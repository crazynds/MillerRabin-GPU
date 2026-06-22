// carry_norm.cu — Carry normalization (propagation) of the raw INTT coefficients
// and the associated big-integer additions. Separated from bigint_ntt.cu because it
// groups the 4 selectable carry algorithms (CARRY_NORM_ALG) and the BigIntNTTBatch
// methods that launch them. The NTT transform kernels themselves live in bigint_ntt.cu.
//
// All limb-touching kernels are templated on the limb storage type `T` (= LimbT):
// `double` for FFT backends, `Data64` for NTT backends (see ops/limb_storage.cuh).
// Limb arithmetic is always done in `uint64_t` internally; only the boundary
// load/store crosses the double↔int line, via limb_ld/limb_st (exact in range).

#include "config.h"
#include "ops/mul/multiplier.cuh"
#include <cstdint>
#include <cstdio>

// ── addition kernels (carry support) ──────────────────────────────────────────

template <typename T>
__global__ static void vadd_batch(
    T *__restrict__ d_c,
    const T *__restrict__ d_a,
    const T *__restrict__ d_b,
    int n, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n)
        return;
    size_t idx = (size_t)cand * n + j;
    limb_st(d_c[idx], limb_ld(d_a[idx]) + limb_ld(d_b[idx]));
}

#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE || CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE || CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
// Add d_raw (stride=padded, raw INTT) into d_dst (stride=n_dst) element-wise.
template <typename T>
__global__ static void vadd_from_raw_batch(
    T *__restrict__ d_dst,
    const T *__restrict__ d_raw,
    int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_dst)
        return;
    if (j < padded)
    {
        size_t di = (size_t)cand * n_dst + j;
        limb_st(d_dst[di], limb_ld(d_dst[di]) + limb_ld(d_raw[(size_t)cand * padded + j]));
    }
}
#endif

// ── Carry normalization algorithms ───────────────────────────────────────────
//
// Select via CARRY_NORM_ALG in config.h:
//   CARRY_ALG_SINGLE_TILE — 1 block/candidate, CARRY_TILE threads, shared-mem carry
//   CARRY_ALG_MULTI_TILE  — parallel intra-tile + parallel single-hop + sequential
//                           residual cleanup (3 kernels)
//   CARRY_ALG_SEQUENTIAL  — 1 thread/candidate, pure sequential loop
//   CARRY_ALG_PREFIX_SCAN — 1 block/candidate, PSCAN_TILE threads, carry-lookahead

// ── CARRY_ALG_SINGLE_TILE ────────────────────────────────────────────────────
// 1 block per candidate, CARRY_TILE threads. Uses shared memory for carry
// propagation — works for any CARRY_TILE (not limited to one warp).
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE

static constexpr int CARRY_TILE = MR_CARRY_TILE;
static_assert(CARRY_TILE >= 32 && (CARRY_TILE % 32) == 0,
              "CARRY_ALG_SINGLE_TILE requires CARRY_TILE to be a multiple of 32");

#ifdef MR_ADVANCED_MONITOR
__device__ unsigned long long g_for_count = 0;
__device__ unsigned long long g_dowhile_count = 0;
#endif

template <typename T>
__global__ static void carry_16bits(
    T *d_src,
    T *d_dst,
    int n, int src_stride, int n_batch)
{
    int tid = threadIdx.x;
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int src_offset = cand * src_stride;
    int dst_offset = cand * n;

    uint64_t tile_carry = 0;
#ifdef MR_ADVANCED_MONITOR
    unsigned long long local_for = 0, local_dowhile = 0;
#endif

#if MR_CARRY_TILE == 32
    // ── Warp path: shuffle intrinsics, no shared memory, no barriers ──────────
    for (int tile = tid; tile < n; tile += CARRY_TILE)
    {
#ifdef MR_ADVANCED_MONITOR
        if (tid == 0)
            local_for++;
#endif
        uint64_t currVal = limb_ld(d_src[src_offset + tile]);
        uint64_t c = (tid == 0) ? tile_carry : 0ULL;
        uint64_t escape = 0;

        unsigned ballot;
        do
        {
#ifdef MR_ADVANCED_MONITOR
            if (tid == 0)
                local_dowhile++;
#endif
            c += currVal;
            currVal = c & LIMB_MASK;
            c >>= LIMB_BITS;

            escape += __shfl_sync(0xFFFFFFFFu, c, CARRY_TILE - 1);
            c = (tid == CARRY_TILE - 1) ? 0 : c;

            uint64_t from_left = __shfl_up_sync(0xFFFFFFFFu, c, 1);
            c = (tid > 0) ? from_left : 0ULL;

            ballot = __ballot_sync(0xFFFFFFFFu, c > 0);
        } while (ballot);

        tile_carry = escape;
        limb_st(d_dst[dst_offset + tile], currVal);
    }
#else
    // ── Block path: shared memory, works for any CARRY_TILE > 32 ─────────────
    __shared__ uint64_t s_carry[CARRY_TILE];
    __shared__ int s_has_carry[2];
    int hc_idx = 0;

    for (int tile = tid; tile < n; tile += CARRY_TILE)
    {
#ifdef MR_ADVANCED_MONITOR
        local_for++;
#endif
        uint64_t currVal = limb_ld(d_src[src_offset + tile]);
        uint64_t c = (tid == 0) ? tile_carry : 0ULL;
        uint64_t escape = 0;

        do
        {
#ifdef MR_ADVANCED_MONITOR
            local_dowhile++;
#endif
            hc_idx ^= 1;
            c += currVal;
            currVal = c & LIMB_MASK;
            c >>= LIMB_BITS;

            s_carry[tid] = c;
            if (tid == 0)
                s_has_carry[hc_idx] = 0;
            __syncthreads();

            escape += s_carry[CARRY_TILE - 1];
            c = (tid > 0) ? s_carry[tid - 1] : 0ULL;

            if (c > 0)
                s_has_carry[hc_idx] = 1;
            __syncthreads();

        } while (s_has_carry[hc_idx]);

        tile_carry = escape;
        limb_st(d_dst[dst_offset + tile], currVal);
    }
#endif

#ifdef MR_ADVANCED_MONITOR
    if (tid == 0)
    {
        atomicAdd(&g_for_count, local_for);
        atomicAdd(&g_dowhile_count, local_dowhile);
    }
#endif
}

// ── CARRY_ALG_MULTI_TILE ─────────────────────────────────────────────────────
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// Phase 1 — copies src→dst and normalizes intra-tile carries in parallel.
// Each block (tile, cand) processes CARRY_TILE elements independently.
// The carry that escapes the tile is saved in d_tile_carry[cand*n_tiles + tile].
template <typename T>
__global__ static void carry_intra_copy(
    T *__restrict__ d_dst,
    const T *__restrict__ d_src,
    Data64 *__restrict__ d_tile_carry,
    int n_dst, int n_src, int n_batch)
{
    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int j_start = tile * CARRY_TILE;
    int j = j_start + tid;

#if MR_CARRY_TILE == 32
    // ── Warp path: same shuffle/ballot trick as carry_16bits, no shared mem ───
    // This tile carries no incoming carry (inter-tile carries are propagated by
    // the second kernel), so we only normalize within the warp and export the
    // carry that escapes lane 31 into d_tile_carry.
    uint64_t currVal = (j < n_src) ? limb_ld(d_src[cand * n_src + j]) : 0ULL;
    uint64_t c = 0;
    uint64_t escape = 0;

    unsigned ballot;
    do
    {
        c += currVal;
        currVal = c & LIMB_MASK;
        c >>= LIMB_BITS;

        escape += __shfl_sync(0xFFFFFFFFu, c, CARRY_TILE - 1);
        c = (tid == CARRY_TILE - 1) ? 0 : c;

        uint64_t from_left = __shfl_up_sync(0xFFFFFFFFu, c, 1);
        c = (tid > 0) ? from_left : 0ULL;

        ballot = __ballot_sync(0xFFFFFFFFu, c > 0);
    } while (ballot);

    if (j < n_dst)
        limb_st(d_dst[cand * n_dst + j], currVal);
    if (tid == 0)
        d_tile_carry[cand * n_tiles + tile] = escape;
#else
    // ── Block path: shared memory, works for any CARRY_TILE > 32 ──────────────
    // Same scheme as carry_16bits: currVal and the scalar carry live in
    // registers, the carry is shuffled to the right neighbour through s_carry[],
    // and the carry escaping lane CARRY_TILE-1 is accumulated into escape.
    __shared__ uint64_t s_carry[CARRY_TILE];
    __shared__ int s_has_carry[2];
    int hc_idx = 0;

    uint64_t currVal = (j < n_src) ? limb_ld(d_src[cand * n_src + j]) : 0ULL;
    uint64_t c = 0;
    uint64_t escape = 0;

    do
    {
        hc_idx ^= 1;
        c += currVal;
        currVal = c & LIMB_MASK;
        c >>= LIMB_BITS;

        s_carry[tid] = c;
        if (tid == 0)
            s_has_carry[hc_idx] = 0;
        __syncthreads();

        escape += s_carry[CARRY_TILE - 1];
        c = (tid > 0) ? s_carry[tid - 1] : 0ULL;

        if (c > 0)
            s_has_carry[hc_idx] = 1;
        __syncthreads();

    } while (s_has_carry[hc_idx]);

    if (j < n_dst)
        limb_st(d_dst[cand * n_dst + j], currVal);
    if (tid == 0)
        d_tile_carry[cand * n_tiles + tile] = escape;
#endif
}

// Phase 2 — parallel single-hop inter-tile propagation (1 thread per receiver tile).
// Thread `t` (1 <= t < n_tiles) reads the escape carry that tile t-1 produced in
// phase 1 (d_tile_carry[t-1]), injects it into the head of its own tile, and
// overwrites that same slot d_tile_carry[t-1] with the residual carry that escapes
// tile t (the part that still needs to reach tile t+1, almost always 0).
//
// Race-free with a single buffer: slot t-1 is read and written only by thread t,
// and tile t's limbs in d_dst are touched only by thread t. No barrier and no
// block-size limit. The last tile's own escape (index n_tiles-1) is left
// untouched as the overall overflow.
template <typename T>
__global__ static void carry_propagate_tiles(
    T *__restrict__ d_dst,
    Data64 *__restrict__ d_tile_carry,
    int n, int n_batch)
{
    int cand = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x + 1; // receiver tile, 1..n_tiles-1
    if (cand >= n_batch)
        return;
    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    if (t >= n_tiles)
        return;

    uint64_t c = (uint64_t)d_tile_carry[cand * n_tiles + (t - 1)];
    if (c != 0)
    {
        int j_start = t * CARRY_TILE;
        int j_end = min(j_start + CARRY_TILE, n);
        for (int j = j_start; c > 0 && j < j_end; j++)
        {
            uint64_t v = limb_ld(d_dst[cand * n + j]) + c;
            limb_st(d_dst[cand * n + j], v & LIMB_MASK);
            c = v >> LIMB_BITS;
        }
    }
    // c is now the residual that escapes tile t (and must enter tile t+1); store
    // it back into the slot we just consumed (0 when nothing escapes).
    d_tile_carry[cand * n_tiles + (t - 1)] = (Data64)c;
}

// Phase 3 — sequential cleanup of the residual carries left by phase 2 (1 thread
// per candidate). After phase 2 each tile has already absorbed the previous tile's
// escape, so the only carries left are the residuals in d_tile_carry, almost
// always zero; this pass is the safety net for the rare multi-tile cascade. The
// residual escaping tile t sits in d_tile_carry[t-1] and enters tile t+1, so tile
// m consumes slot m-2.
template <typename T>
__global__ static void carry_inter_tiles(
    T *__restrict__ d_dst,
    Data64 *__restrict__ d_tile_carry,
    int n, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    uint64_t r = 0; // running carry cascading across tiles in this cleanup pass
    for (int m = 2; m < n_tiles; m++)
    {
        uint64_t c = r + (uint64_t)d_tile_carry[cand * n_tiles + (m - 2)];
        r = 0;
        if (c == 0)
            continue; // nothing left to push into tile m
        int j_start = m * CARRY_TILE;
        int j_end = min(j_start + CARRY_TILE, n);
        for (int j = j_start; c > 0 && j < j_end; j++)
        {
            uint64_t v = limb_ld(d_dst[cand * n + j]) + c;
            limb_st(d_dst[cand * n + j], v & LIMB_MASK);
            c = v >> LIMB_BITS;
        }
        r = c; // escapes tile m, will enter tile m+1 on the next iteration
    }
}

// ── CARRY_ALG_SEQUENTIAL ─────────────────────────────────────────────────────
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL

// 1 thread per candidate — copies d_src (stride=n_src) → d_dst (stride=n_dst)
// normalizing carries sequentially.
template <typename T>
__global__ static void carry_sequential(
    T *__restrict__ d_dst,
    const T *__restrict__ d_src,
    int n_dst, int n_src, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    uint64_t carry = 0;
    for (int j = 0; j < n_dst; j++)
    {
        uint64_t v = (j < n_src ? limb_ld(d_src[cand * n_src + j]) : 0ULL) + carry;
        limb_st(d_dst[cand * n_dst + j], v & LIMB_MASK);
        carry = v >> LIMB_BITS;
    }
}

// Fused version for add_raw_buf_and_carry: adds d_raw (raw INTT, stride=padded)
// into d_dst and normalizes carries in a single sequential pass per candidate.
template <typename T>
__global__ static void vadd_carry_sequential(
    T *__restrict__ d_dst,
    const T *__restrict__ d_raw,
    int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    uint64_t carry = 0;
    for (int j = 0; j < n_dst; j++)
    {
        uint64_t raw = (j < padded ? limb_ld(d_raw[cand * padded + j]) : 0ULL);
        uint64_t v = limb_ld(d_dst[cand * n_dst + j]) + raw + carry;
        limb_st(d_dst[cand * n_dst + j], v & LIMB_MASK);
        carry = v >> LIMB_BITS;
    }
}

// ── CARRY_ALG_PREFIX_SCAN ────────────────────────────────────────────────────
// Carry-lookahead adder (Kogge-Stone). 1 block per candidate, PSCAN_TILE threads.
//
// Idea: a raw 64-bit limb is the sum of 4 "planes" of 16-bit digits,
// where plane k contributes to digit (i+k):
//     v[i] = c0(i) + c1(i)·2^16 + c2(i)·2^32 + c3(i)·2^48
//   ⇒ value = Σ c_k(i)·2^(16(i+k))   ⇒  plane k holds c_k(i) at digit i+k.
// Normalizing = adding the 4 shifted planes. We do it via the tree
//     (p0 + p1) + (p2 + p3)
// and each addition is of two numbers ALREADY normalized in base 2^16, so the carry
// of each digit is at most 1 (binary). This enables the generate/propagate algebra:
//     gen[j]  = (A[j]+B[j]) >= 2^16            (the position generates a carry)
//     prop[j] = (A[j]+B[j]) == 2^16 - 1        (the position propagates an incoming carry)
// The carry entering each position comes from a prefix-scan (Kogge-Stone) of the
// associative operator  (G,P) = (G_hi | (P_hi & G_lo),  P_hi & P_lo),  in log(T) steps —
// without a sequential propagation pass.
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN

// This algorithm decomposes the raw 64-bit limb into 4 FIXED 16-bit "planes"
// (>>16, >>32, >>48) — only valid for LIMB_BITS == 16. The other carry algorithms
// (SINGLE_TILE/MULTI_TILE/SEQUENTIAL) are parametric in LIMB_BITS.
#if LIMB_BITS != 16
#error "CARRY_ALG_PREFIX_SCAN requires LIMB_BITS == 16 (decomposition into 16-bit planes). Use another CARRY_NORM_ALG for LIMB_BITS != 16."
#endif

static constexpr int PSCAN_TILE = MR_CARRY_TILE;
static_assert(PSCAN_TILE >= 32 && PSCAN_TILE <= 1024 && (PSCAN_TILE % 32) == 0,
              "MR_CARRY_TILE must be a multiple of 32 between 32 and 1024");

static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#if PSCAN_TILE == 32
// ───── 1-warp path (PSCAN_TILE == 32): shuffle only, no shared, no barrier ──────

// Carry-lookahead addition of two already-normalized digits, within a single warp.
// `s` = A[lane] + B[lane] (≤ 2·LIMB_MASK, binary carry). `cin` is the scalar carry
// entering the tile (identical across all lanes). Returns the lane's normalized
// digit; writes into *tile_cout the carry leaving the tile (identical across all lanes).
// All via __shfl — no __syncthreads and no shared memory.
__device__ static inline uint64_t cla_warp(uint64_t s, int cin, int *tile_cout)
{
    int lane = threadIdx.x;                                // 0..31
    unsigned g = (unsigned)((s >> LIMB_BITS) & 1ULL);      // generate
    unsigned p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u; // propagate

    // Kogge-Stone via shuffle: inclusive scan of the operator (G,P), low→high.
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        unsigned gl = __shfl_up_sync(FULL_MASK, g, d);
        unsigned pl = __shfl_up_sync(FULL_MASK, p, d);
        if (lane >= d)
        {
            g = g | (p & gl);
            p = p & pl;
        }
    }
    // (g,p) = inclusive aggregate of [0..lane]. Carry ENTERING the lane = EXCLUSIVE
    // aggregate [0..lane-1] (shuffle ↑1) applied to cin; an empty prefix propagates cin.
    unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
    unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
    if (lane == 0)
    {
        eg = 0u;
        ep = 1u;
    }
    int carry_in = (int)(eg | (ep & (unsigned)cin));

    // Carry LEAVING the tile = inclusive aggregate of the whole warp (lane 31) with cin.
    unsigned Gt = __shfl_sync(FULL_MASK, g, 31);
    unsigned Pt = __shfl_sync(FULL_MASK, p, 31);
    *tile_cout = (int)(Gt | (Pt & (unsigned)cin));

    return ((s & LIMB_MASK) + (uint64_t)carry_in) & LIMB_MASK;
}

// Normalizes d_src (raw 64-bit limbs, stride=src_stride) → d_dst (stride=n_dst)
// in base 2^16. 1 warp per candidate. Safe in-place (src == dst, src_stride ==
// n_dst): the 3-limb halo of the previous tile stays in registers (pr1/pr2/pr3),
// and each lane only writes its own position after already having read the raw limb.
template <typename T>
__global__ static void pscan_normalize(
    T *__restrict__ d_dst,
    const T *__restrict__ d_src,
    int n_dst, int src_stride, int n_batch)
{
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int lane = threadIdx.x;
    const T *src = d_src + (size_t)cand * src_stride;
    T *dst = d_dst + (size_t)cand * n_dst;

    int cin1 = 0, cin2 = 0, cin3 = 0; // scalar carries between tiles
    // Halo: global raw limbs base-1, base-2, base-3 (0 before the first tile).
    uint64_t pr1 = 0, pr2 = 0, pr3 = 0;

#ifdef MR_ADVANCED_MONITOR
    unsigned long long local_for = 0, local_dowhile = 0;
#endif

    for (int base = 0; base < n_dst; base += PSCAN_TILE)
    {
        int j = base + lane;
        uint64_t raw0 = (j < n_dst) ? limb_ld(src[j]) : 0ULL; // position j
#ifdef MR_ADVANCED_MONITOR
        if (lane == 0)
        {
            local_for++;
            local_dowhile += 3;
        }
#endif
        // Halo inside the tile via shuffle; outside the tile (lane < k) comes from pr1/pr2/pr3.
        uint64_t sh1 = __shfl_up_sync(FULL_MASK, raw0, 1);
        uint64_t sh2 = __shfl_up_sync(FULL_MASK, raw0, 2);
        uint64_t sh3 = __shfl_up_sync(FULL_MASK, raw0, 3);
        uint64_t raw1 = (lane >= 1) ? sh1 : pr1;                                         // position j-1
        uint64_t raw2 = (lane >= 2) ? sh2 : (lane == 1 ? pr1 : pr2);                     // position j-2
        uint64_t raw3 = (lane >= 3) ? sh3 : (lane == 2 ? pr1 : (lane == 1 ? pr2 : pr3)); // j-3

        uint64_t c0 = raw0 & LIMB_MASK;         // plane 0, digit j
        uint64_t c1 = (raw1 >> 16) & LIMB_MASK; // plane 1, shifted +1
        uint64_t c2 = (raw2 >> 32) & LIMB_MASK; // plane 2, shifted +2
        uint64_t c3 = (raw3 >> 48) & LIMB_MASK; // plane 3, shifted +3

        int cout;
        uint64_t r1 = cla_warp(c0 + c1, cin1, &cout);
        cin1 = cout;
        uint64_t r2 = cla_warp(c2 + c3, cin2, &cout);
        cin2 = cout;
        uint64_t digit = cla_warp(r1 + r2, cin3, &cout);
        cin3 = cout;

        if (j < n_dst)
            limb_st(dst[j], digit);

        // Halo of the next tile = raw limbs of lanes 31,30,29 of this tile.
        pr1 = __shfl_sync(FULL_MASK, raw0, 31);
        pr2 = __shfl_sync(FULL_MASK, raw0, 30);
        pr3 = __shfl_sync(FULL_MASK, raw0, 29);
    }

#ifdef MR_ADVANCED_MONITOR
    if (lane == 0)
    {
        atomicAdd(&g_for_count, local_for);
        atomicAdd(&g_dowhile_count, local_dowhile);
    }
#endif
}

#else
// ───── Block-wide path (PSCAN_TILE > 32): hierarchical warp→block scan ──────────
// Inclusive scan within each warp via shuffle (no barrier), then combines the
// NWARPS warp totals in shared memory (1 __syncthreads) and applies the offset.
// Halo neighbors cross warps, so the tile's raw limbs go to shared memory.

static constexpr int PSCAN_NWARPS = PSCAN_TILE / 32;

// Carry-lookahead addition of two already-normalized digits, cooperative across the whole block.
// wG/wP are shared memory buffers of size PSCAN_NWARPS (per-warp totals).
__device__ static uint64_t cla_block(uint64_t s, int cin,
                                     unsigned *wG, unsigned *wP, int *tile_cout)
{
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    unsigned g = (unsigned)((s >> LIMB_BITS) & 1ULL);      // generate
    unsigned p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u; // propagate

    // 1) inclusive scan (G,P) within the warp, via shuffle.
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        unsigned gl = __shfl_up_sync(FULL_MASK, g, d);
        unsigned pl = __shfl_up_sync(FULL_MASK, p, d);
        if (lane >= d)
        {
            g = g | (p & gl);
            p = p & pl;
        }
    }
    // intra-warp exclusive (aggregate of [warpbase .. t-1]) via shuffle ↑1.
    unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
    unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
    if (lane == 0)
    {
        eg = 0u;
        ep = 1u;
    }

    // 2) publish each warp's total (lane 31 = inclusive aggregate of the warp).
    if (lane == 31)
    {
        wG[warp] = g;
        wP[warp] = p;
    }
    __syncthreads();

    // 3) EXCLUSIVE prefix across warps [0..warp-1] (fold in low→high order) and total.
    unsigned Gpre = 0u, Ppre = 1u; // aggregate of warps less significant than `warp`
    unsigned Gtot = 0u, Ptot = 1u; // aggregate of all warps
#pragma unroll
    for (int w = 0; w < PSCAN_NWARPS; w++)
    {
        unsigned gw = wG[w], pw = wP[w];
        if (w < warp)
        {
            Gpre = gw | (pw & Gpre);
            Ppre = pw & Ppre;
        }
        Gtot = gw | (pw & Gtot);
        Ptot = pw & Ptot;
    }

    // 4) carry ENTERING position t = combine(warp-prefix, intra-warp-exclusive)
    //    applied to cin. low = warp prefix; high = intra-warp exclusive.
    unsigned Cg = eg | (ep & Gpre);
    unsigned Cp = ep & Ppre;
    int carry_in = (int)(Cg | (Cp & (unsigned)cin));

    // carry LEAVING the tile = aggregate of the whole block applied to cin.
    *tile_cout = (int)(Gtot | (Ptot & (unsigned)cin));

    __syncthreads(); // wG/wP will be reused on the next call
    return ((s & LIMB_MASK) + (uint64_t)carry_in) & LIMB_MASK;
}

template <typename T>
__global__ static void pscan_normalize(
    T *__restrict__ d_dst,
    const T *__restrict__ d_src,
    int n_dst, int src_stride, int n_batch)
{
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int t = threadIdx.x;
    const T *src = d_src + (size_t)cand * src_stride;
    T *dst = d_dst + (size_t)cand * n_dst;

    __shared__ uint64_t sraw[PSCAN_TILE];
    __shared__ unsigned wG[PSCAN_NWARPS];
    __shared__ unsigned wP[PSCAN_NWARPS];
    __shared__ uint64_t sprev[3]; // raw limbs of positions base-1, base-2, base-3

    if (t < 3)
        sprev[t] = 0ULL;
    int cin1 = 0, cin2 = 0, cin3 = 0; // scalar carries between tiles

    for (int base = 0; base < n_dst; base += PSCAN_TILE)
    {
        int j = base + t;
        sraw[t] = (j < n_dst) ? limb_ld(src[j]) : 0ULL;
        __syncthreads();
        // Halo with neighbors in the tile (shared) or in the previous tile (sprev).
        uint64_t raw0 = sraw[t];                                          // position j
        uint64_t raw1 = (t >= 1) ? sraw[t - 1] : sprev[0];                // position j-1
        uint64_t raw2 = (t >= 2) ? sraw[t - 2] : sprev[(t == 0) ? 1 : 0]; // j-2
        uint64_t raw3 = (t >= 3) ? sraw[t - 3] : sprev[2 - t];            // j-3

        uint64_t c0 = raw0 & LIMB_MASK;         // plane 0, digit j
        uint64_t c1 = (raw1 >> 16) & LIMB_MASK; // plane 1, shifted +1
        uint64_t c2 = (raw2 >> 32) & LIMB_MASK; // plane 2, shifted +2
        uint64_t c3 = (raw3 >> 48) & LIMB_MASK; // plane 3, shifted +3

        int cout;
        uint64_t r1 = cla_block(c0 + c1, cin1, wG, wP, &cout);
        cin1 = cout;
        uint64_t r2 = cla_block(c2 + c3, cin2, wG, wP, &cout);
        cin2 = cout;
        uint64_t digit = cla_block(r1 + r2, cin3, wG, wP, &cout);
        cin3 = cout;

        if (j < n_dst)
            limb_st(dst[j], digit);

        // Preserve the last 3 raw limbs of the tile for the next one's halo.
        __syncthreads();
        if (t == 0)
        {
            sprev[0] = sraw[PSCAN_TILE - 1];
            sprev[1] = sraw[PSCAN_TILE - 2];
            sprev[2] = sraw[PSCAN_TILE - 3];
        }
        __syncthreads();
    }
}

#endif // PSCAN_TILE == 32

#else
#error "CARRY_NORM_ALG must be CARRY_ALG_SINGLE_TILE, CARRY_ALG_MULTI_TILE, CARRY_ALG_SEQUENTIAL or CARRY_ALG_PREFIX_SCAN"
#endif

// ── Advanced monitor: global carry stats ─────────────────────────────────────

void carry_stats_print_and_reset()
{
#ifdef MR_ADVANCED_MONITOR
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    unsigned long long h_for, h_dowhile;
    cudaMemcpyFromSymbol(&h_for, g_for_count, sizeof(h_for));
    cudaMemcpyFromSymbol(&h_dowhile, g_dowhile_count, sizeof(h_dowhile));
    if (h_for > 0)
        printf("[carry_16bits] for=%llu  do-while=%llu  mean=%.3f iter/tile\n",
               h_for, h_dowhile, (double)h_dowhile / (double)h_for);
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(g_for_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_dowhile_count, &zero, sizeof(zero));
#endif
#endif
}

// ── Multiplier methods that launch the carry kernels ──────────────────────────
// d_buf_A holds the raw transformed/INTT coefficients in the limb storage type:
// reinterpret it as LimbT* (identity when LimbT == Data64; byte-reinterpret of the
// real INTT output when LimbT == double).

void Multiplier::vadd_raw_buf(LimbT *d_dst, int n_dst, cudaStream_t s)
{
    LimbT *raw = raw_coeffs();
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    vadd_from_raw_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, raw, n_dst, padded, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE || CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n_dst + THR - 1) / THR;
    vadd_from_raw_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_dst, raw, n_dst, padded, n_batch);
#endif
    // SEQUENTIAL: has no separate vadd — use add_raw_buf_and_carry directly
}

void Multiplier::carry_to_limbs(LimbT *d_out, int n_out, cudaStream_t s)
{
    LimbT *raw = raw_coeffs();
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_out + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_out, raw, d_tile_carry, n_out, padded, n_batch);
    if (n_tiles > 1)
        carry_propagate_tiles<<<dim3((n_tiles - 1 + THR - 1) / THR, n_batch), THR, 0, s>>>(
            d_out, d_tile_carry, n_out, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_out, d_tile_carry, n_out, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(raw, d_out, n_out, padded, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    int blk = (n_batch + CARRY_TILE - 1) / CARRY_TILE;
    carry_sequential<<<blk, CARRY_TILE, 0, s>>>(d_out, raw, n_out, padded, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    pscan_normalize<<<n_batch, PSCAN_TILE, 0, s>>>(d_out, raw, n_out, padded, n_batch);
#endif
}

void Multiplier::carry_after_vadd(LimbT *d_dst, int n_dst, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_dst, d_tile_carry, n_dst, n_dst, n_batch);
    if (n_tiles > 1)
        carry_propagate_tiles<<<dim3((n_tiles - 1 + THR - 1) / THR, n_batch), THR, 0, s>>>(
            d_dst, d_tile_carry, n_dst, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_dst, d_tile_carry, n_dst, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_dst, d_dst, n_dst, n_dst, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    pscan_normalize<<<n_batch, PSCAN_TILE, 0, s>>>(d_dst, d_dst, n_dst, n_dst, n_batch);
#endif
    // SEQUENTIAL: no-op — carry was already done in add_raw_buf_and_carry
}

void Multiplier::add_raw_buf_and_carry(LimbT *d_dst, int n_dst,
                                       cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE || CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE || CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    vadd_raw_buf(d_dst, n_dst, s);
    carry_after_vadd(d_dst, n_dst, s);
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    LimbT *raw = raw_coeffs();
    int blk = (n_batch + CARRY_TILE - 1) / CARRY_TILE;
    vadd_carry_sequential<<<blk, CARRY_TILE, 0, s>>>(d_dst, raw, n_dst, padded, n_batch);
#endif
}

void Multiplier::add_and_carry(LimbT *d_a, const LimbT *d_b, int n, int n_passes,
                               cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    vadd_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_tile_carry, n, n, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_a, d_tile_carry, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_a, d_a, n, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    // carry_sequential reads d_src and writes d_dst; in-place is safe (j grows)
    // But first we need to add d_b into d_a
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    int blk = (n_batch + CARRY_TILE - 1) / CARRY_TILE;
    carry_sequential<<<blk, CARRY_TILE, 0, s>>>(d_a, d_a, n, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    pscan_normalize<<<n_batch, PSCAN_TILE, 0, s>>>(d_a, d_a, n, n, n_batch);
#endif
}
