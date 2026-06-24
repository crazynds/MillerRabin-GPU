// ops/sub/sub.cu — kernels and launchers for tiled subtraction (prefix-scan G/P/K).
#include "ops/sub/sub.cuh"
#include "config.h" // MR_SUB_TILE

namespace
{
    constexpr int CS_TILE = MR_SUB_TILE;

    // Composition of borrow G/P/K states: state = bw0 | (bw1 << 1).
    __device__ int cs_combine(int L, int R)
    {
        int c0 = (R >> (L & 1)) & 1;
        int c1 = (R >> ((L >> 1) & 1)) & 1;
        return c0 | (c1 << 1);
    }

    // Phase 1: per tile, comparison a vs b and the tile's borrow G/P/K state.
    template <typename T>
    __global__ void sub_phase1_k(
        const T *__restrict__ a, int sa,
        const T *__restrict__ b, int sb,
        const int *__restrict__ bk, int W,
        int *__restrict__ tile_cmp, int *__restrict__ tile_bstate, int n_batch)
    {
        __shared__ uint64_t s_a[CS_TILE];
        __shared__ uint64_t s_b[CS_TILE];
        __shared__ int s_reduce[CS_TILE];
        __shared__ int s_state[CS_TILE];

        int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
        int j = tile * CS_TILE + tid;
        int n_tiles = (W + CS_TILE - 1) / CS_TILE;
        if (cand >= n_batch)
            return;

        int bw = bk ? bk[cand] : W;
        s_a[tid] = (j < W) ? limb_ld(a[(size_t)cand * sa + j]) : 0ULL;
        s_b[tid] = (j < W && j < bw) ? limb_ld(b[(size_t)cand * sb + j]) : 0ULL;
        __syncthreads();

        int enc = 1;
        if (j < W && s_a[tid] != s_b[tid])
            enc = ((j + 1) << 2) | ((s_a[tid] > s_b[tid]) ? 2 : 0);
        s_reduce[tid] = enc;
        __syncthreads();
        for (int stride = CS_TILE >> 1; stride > 0; stride >>= 1)
        {
            if (tid < stride && s_reduce[tid + stride] > s_reduce[tid])
                s_reduce[tid] = s_reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0)
            tile_cmp[cand * n_tiles + tile] = (s_reduce[0] == 1) ? 0 : ((s_reduce[0] & 3) - 1);

        int bw0 = (j < W && s_a[tid] < s_b[tid]) ? 1 : 0;
        int bw1 = (j < W && s_a[tid] <= s_b[tid]) ? 1 : 0;
        s_state[tid] = bw0 | (bw1 << 1);
        __syncthreads();
        for (int stride = 1; stride < CS_TILE; stride <<= 1)
        {
            int combined = (tid >= stride) ? cs_combine(s_state[tid - stride], s_state[tid])
                                           : s_state[tid];
            __syncthreads();
            s_state[tid] = combined;
            __syncthreads();
        }
        int last = min(CS_TILE, W - tile * CS_TILE) - 1;
        if (tid == last)
            tile_bstate[cand * n_tiles + tile] = s_state[tid];
    }

    // Phase 2 (fused with resolve): resolves the tile's borrow_in and applies out = a − b.
    template <typename T>
    __global__ void sub_apply_k(
        T *__restrict__ out, int so,
        const T *__restrict__ a, int sa,
        const T *__restrict__ b, int sb,
        const int *__restrict__ bk, int W,
        const int *__restrict__ tile_cmp, const int *__restrict__ tile_bstate,
        int uncond, int n_batch)
    {
        __shared__ uint64_t s_a[CS_TILE];
        __shared__ uint64_t s_b[CS_TILE];
        __shared__ int s_state[CS_TILE];
        __shared__ int s_tile_bin;

        int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
        int j = tile * CS_TILE + tid;
        int n_tiles = (W + CS_TILE - 1) / CS_TILE;
        if (cand >= n_batch)
            return;

        if (tid == 0)
        {
            const int *cmp = tile_cmp + cand * n_tiles;
            const int *bstate = tile_bstate + cand * n_tiles;
            int do_sub = 1;
            if (!uncond)
            {
                int gcmp = 0;
                for (int t = n_tiles - 1; t >= 0 && gcmp == 0; t--)
                    gcmp = cmp[t];
                if (gcmp < 0)
                    do_sub = 0; // a < b
            }
            int bin = -1;
            if (do_sub)
            {
                int cur = 0;
                for (int t = 0; t < tile; t++)
                    cur = (bstate[t] >> cur) & 1;
                bin = cur;
            }
            s_tile_bin = bin;
        }
        __syncthreads();

        int tile_bin_v = s_tile_bin;
        if (tile_bin_v < 0)
            return; // a < b: no-op (out already contains a in the in-place case)

        int bw = bk ? bk[cand] : W;
        s_a[tid] = (j < W) ? limb_ld(a[(size_t)cand * sa + j]) : 0ULL;
        s_b[tid] = (j < W && j < bw) ? limb_ld(b[(size_t)cand * sb + j]) : 0ULL;
        __syncthreads();

        int bw0 = (j < W && s_a[tid] < s_b[tid]) ? 1 : 0;
        int bw1 = (j < W && s_a[tid] <= s_b[tid]) ? 1 : 0;
        s_state[tid] = bw0 | (bw1 << 1);
        __syncthreads();
        for (int stride = 1; stride < CS_TILE; stride <<= 1)
        {
            int combined = (tid >= stride) ? cs_combine(s_state[tid - stride], s_state[tid])
                                           : s_state[tid];
            __syncthreads();
            s_state[tid] = combined;
            __syncthreads();
        }
        int prefix_excl = (tid == 0) ? 2 : s_state[tid - 1];
        int bin = (prefix_excl >> tile_bin_v) & 1;

        if (j < W)
        {
            int64_t d = (int64_t)s_a[tid] - (int64_t)s_b[tid] - bin;
            limb_st(out[(size_t)cand * so + j], (uint64_t)((d < 0) ? d + (1LL << LIMB_BITS) : d));
        }
    }

    template <typename T>
    __global__ void copy_low_k(T *__restrict__ out, const T *__restrict__ r,
                               int out_limbs, int W, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= out_limbs)
            return;
        out[(size_t)cand * out_limbs + j] = (j < W) ? r[(size_t)cand * W + j] : (T)0;
    }
}

namespace ops
{
    int sub_n_tiles(int W) { return (W + CS_TILE - 1) / CS_TILE; }

    void sub_phase1(const LimbT *a, int sa, const LimbT *b, int sb,
                    const int *bk, int W, int *tile_cmp, int *tile_bstate,
                    int n_batch, cudaStream_t s)
    {
        dim3 g((unsigned)sub_n_tiles(W), (unsigned)n_batch);
        sub_phase1_k<LimbT><<<g, CS_TILE, 0, s>>>(a, sa, b, sb, bk, W, tile_cmp, tile_bstate, n_batch);
    }

    void sub_apply(LimbT *out, int so, const LimbT *a, int sa, const LimbT *b, int sb,
                   const int *bk, int W, const int *tile_cmp, const int *tile_bstate,
                   int uncond, int n_batch, cudaStream_t s)
    {
        dim3 g((unsigned)sub_n_tiles(W), (unsigned)n_batch);
        sub_apply_k<LimbT><<<g, CS_TILE, 0, s>>>(out, so, a, sa, b, sb, bk, W,
                                                 tile_cmp, tile_bstate, uncond, n_batch);
    }

    void copy_low(LimbT *out, const LimbT *r, int out_limbs, int W,
                  int n_batch, cudaStream_t s)
    {
        constexpr int thr = MR_THR_COPY;
        dim3 g((unsigned)(out_limbs + thr - 1) / thr, (unsigned)n_batch);
        copy_low_k<LimbT><<<g, thr, 0, s>>>(out, r, out_limbs, W, n_batch);
    }
}
