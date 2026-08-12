/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/sub/sub.cu
 * ROLE   batched tiled big-integer subtraction
 *
 * HOW    Three kernels. phase1: one block per (candidate, tile) folds the
 *        tile borrow G/P/K state and, when asked, the a-vs-b comparison.
 *        resolve: one thread per candidate turns the per-tile states into
 *        each tile borrow_in. apply: subtracts with that borrow_in.
 *
 * NOTE   The scans are warp-level (__shfl_up_sync), one barrier per block.
 *        The shared-memory version cost about 24 barriers and was
 *        insensitive to bytes moved — halving its traffic changed its time
 *        by 1%, which is what identified it as barrier-bound rather than
 *        bandwidth-bound.
 *
 * CHANGELOG
 *   2026-08-11  Rewrote the scans at warp level, made the comparison
 *               optional, and replaced the per-block serial walk over
 *               preceding tiles (quadratic in tile count) with the resolve
 *               kernel. 113.9 -> 58.9 us on phase1.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "ops/sub/sub.cuh"
#include "config.h"

namespace
{
    constexpr int CS_TILE = MR_SUB_TILE;
    static_assert(CS_TILE >= 32 && (CS_TILE % 32) == 0,
                  "MR_SUB_TILE must be a multiple of 32 (warp-level scans)");
    constexpr int CS_WARPS = CS_TILE / 32;

    // Composition of borrow G/P/K states: state = bw0 | (bw1 << 1), i.e. state
    // encodes f(x) = (state >> x) & 1 — the borrow out of the range given borrow x
    // into it. Associative (so it scans), and f(x) = x is the identity, = 2.
    __device__ __forceinline__ int cs_combine(int L, int R)
    {
        int c0 = (R >> (L & 1)) & 1;
        int c1 = (R >> ((L >> 1) & 1)) & 1;
        return c0 | (c1 << 1);
    }
    constexpr int CS_IDENTITY = 2;

    // Inclusive scan of cs_combine across one warp: 5 shuffles, no barriers.
    __device__ __forceinline__ int warp_scan_state(int v, unsigned lane)
    {
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
            const int n = __shfl_up_sync(0xFFFFFFFFu, v, d);
            if (lane >= (unsigned)d)
                v = cs_combine(n, v);
        }
        return v;
    }

    // Phase 1: per tile, the borrow state and (optionally) the a-vs-b comparison.
    // need_cmp is 0 for unconditional subtractions, where the comparison is dead
    // work — that is one of the three subtractions in a Barrett reduction.
    template <typename T>
    __global__ void sub_phase1_k(
        const T *__restrict__ a, int sa,
        const T *__restrict__ b, int sb,
        const int *__restrict__ bk, int W,
        int *__restrict__ tile_cmp, int *__restrict__ tile_bstate,
        int need_cmp, int n_batch)
    {
        __shared__ int s_warp[CS_WARPS];
        __shared__ int s_red[CS_WARPS];

        const int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
        const int j = tile * CS_TILE + tid;
        const int n_tiles = (W + CS_TILE - 1) / CS_TILE;
        if (cand >= n_batch)
            return;

        const int bw = bk ? bk[cand] : W;
        const uint64_t va = (j < W) ? limb_ld(a[(size_t)cand * sa + j]) : 0ULL;
        const uint64_t vb = (j < W && j < bw) ? limb_ld(b[(size_t)cand * sb + j]) : 0ULL;

        const unsigned lane = tid & 31u;
        const int w = tid >> 5;

        const int state = ((va < vb) ? 1 : 0) | (((va <= vb) ? 1 : 0) << 1);
        const int inc = warp_scan_state(state, lane);
        if (lane == 31u)
            s_warp[w] = inc;

        int enc = 1;
        if (need_cmp)
        {
            if (j < W && va != vb)
                enc = ((j + 1) << 2) | ((va > vb) ? 2 : 0);
#pragma unroll
            for (int d = 16; d > 0; d >>= 1)
                enc = max(enc, __shfl_down_sync(0xFFFFFFFFu, enc, d));
            if (lane == 0u)
                s_red[w] = enc;
        }

        __syncthreads();

        if (tid == 0)
        {
            int total = CS_IDENTITY;
#pragma unroll
            for (int i = 0; i < CS_WARPS; i++)
                total = cs_combine(total, s_warp[i]);
            tile_bstate[cand * n_tiles + tile] = total;

            if (need_cmp)
            {
                int m = s_red[0];
#pragma unroll
                for (int i = 1; i < CS_WARPS; i++)
                    m = max(m, s_red[i]);
                tile_cmp[cand * n_tiles + tile] = (m == 1) ? 0 : ((m & 3) - 1);
            }
        }
    }

    // Resolve: one thread per candidate walks the per-tile states ONCE and writes
    // each tile's borrow_in (or -1 meaning "a < b, skip the subtraction").
    // Previously every apply block re-derived this by looping over all preceding
    // tiles, which is O(n_tiles^2) across the grid and fully serial per block.
    __global__ void sub_resolve_k(
        const int *__restrict__ tile_cmp,
        const int *__restrict__ tile_bstate,
        signed char *__restrict__ tile_bin,
        int n_tiles, int uncond, int n_batch)
    {
        const int cand = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch)
            return;

        const int *cmp = tile_cmp + (size_t)cand * n_tiles;
        const int *bst = tile_bstate + (size_t)cand * n_tiles;
        signed char *out = tile_bin + (size_t)cand * n_tiles;

        if (!uncond)
        {
            int gcmp = 0;
            for (int t = n_tiles - 1; t >= 0 && gcmp == 0; t--)
                gcmp = cmp[t];
            if (gcmp < 0)
            {
                for (int t = 0; t < n_tiles; t++)
                    out[t] = -1;
                return;
            }
        }

        int cur = 0;
        for (int t = 0; t < n_tiles; t++)
        {
            out[t] = (signed char)cur;
            cur = (bst[t] >> cur) & 1;
        }
    }

    // Apply: out = a - b, using the tile's resolved borrow_in.
    template <typename T>
    __global__ void sub_apply_k(
        T *__restrict__ out, int so,
        const T *__restrict__ a, int sa,
        const T *__restrict__ b, int sb,
        const int *__restrict__ bk, int W,
        const signed char *__restrict__ tile_bin, int n_batch)
    {
        __shared__ int s_warp[CS_WARPS];

        const int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
        const int j = tile * CS_TILE + tid;
        const int n_tiles = (W + CS_TILE - 1) / CS_TILE;
        if (cand >= n_batch)
            return;

        const int tile_bin_v = tile_bin[cand * n_tiles + tile];
        if (tile_bin_v < 0)
            return;

        const int bw = bk ? bk[cand] : W;
        const uint64_t va = (j < W) ? limb_ld(a[(size_t)cand * sa + j]) : 0ULL;
        const uint64_t vb = (j < W && j < bw) ? limb_ld(b[(size_t)cand * sb + j]) : 0ULL;

        const unsigned lane = tid & 31u;
        const int w = tid >> 5;

        const int state = ((va < vb) ? 1 : 0) | (((va <= vb) ? 1 : 0) << 1);
        const int inc = warp_scan_state(state, lane);
        if (lane == 31u)
            s_warp[w] = inc;
        __syncthreads();

        int pre = CS_IDENTITY;
        for (int i = 0; i < w; i++)
            pre = cs_combine(pre, s_warp[i]);
        int excl_w = __shfl_up_sync(0xFFFFFFFFu, inc, 1);
        if (lane == 0u)
            excl_w = CS_IDENTITY;
        const int prefix_excl = cs_combine(pre, excl_w);

        const int bin = (prefix_excl >> tile_bin_v) & 1;

        if (j < W)
        {
            const int64_t d = (int64_t)va - (int64_t)vb - bin;
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
                    int need_cmp, int n_batch, cudaStream_t s)
    {
        dim3 g((unsigned)sub_n_tiles(W), (unsigned)n_batch);
        sub_phase1_k<LimbT><<<g, CS_TILE, 0, s>>>(a, sa, b, sb, bk, W,
                                                  tile_cmp, tile_bstate, need_cmp, n_batch);
    }

    void sub_resolve(const int *tile_cmp, const int *tile_bstate, signed char *tile_bin,
                     int W, int uncond, int n_batch, cudaStream_t s)
    {
        constexpr int THR = 128;
        int blk = (n_batch + THR - 1) / THR;
        sub_resolve_k<<<blk, THR, 0, s>>>(tile_cmp, tile_bstate, tile_bin,
                                          sub_n_tiles(W), uncond, n_batch);
    }

    void sub_apply(LimbT *out, int so, const LimbT *a, int sa, const LimbT *b, int sb,
                   const int *bk, int W, const signed char *tile_bin,
                   int n_batch, cudaStream_t s)
    {
        dim3 g((unsigned)sub_n_tiles(W), (unsigned)n_batch);
        sub_apply_k<LimbT><<<g, CS_TILE, 0, s>>>(out, so, a, sa, b, sb, bk, W,
                                                 tile_bin, n_batch);
    }

    void copy_low(LimbT *out, const LimbT *r, int out_limbs, int W,
                  int n_batch, cudaStream_t s)
    {
        constexpr int thr = MR_THR_COPY;
        dim3 g((unsigned)(out_limbs + thr - 1) / thr, (unsigned)n_batch);
        copy_low_k<LimbT><<<g, thr, 0, s>>>(out, r, out_limbs, W, n_batch);
    }
}
