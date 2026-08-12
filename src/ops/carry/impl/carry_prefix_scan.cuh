/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/impl/carry_prefix_scan.cuh
 * ROLE   CARRY_ALG_PREFIX_SCAN — carry-lookahead (Kogge-Stone),
 *        experimental
 *
 * HOW    A raw 64-bit limb is the sum of four 16-bit planes, where plane k
 *        contributes to digit i+k. Normalizing is adding the four shifted
 *        planes as (p0+p1) + (p2+p3); each of those adds two numbers
 *        already normalized in base 2^16, so every digit carries at most 1.
 *        That enables the generate/propagate algebra, and the carry
 *        entering each position comes from a Kogge-Stone prefix scan of
 *        (G,P) = (G_hi | (P_hi & G_lo), P_hi & P_lo) in log(T) steps, with
 *        no sequential pass.
 *
 * NOTE   The fixed 16-bit planes are why this one is locked to LIMB_BITS ==
 *        16, unlike the other three which are parametric. It also cannot
 *        fuse an addend, so it is incompatible with MR_LIMB32 under
 *        Montgomery.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from carry_norm.cu; gained the carry_impl entry
 *               points.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/carry/carry_launch.cuh"
#include "ops/carry/carry_add.cuh"

/* This algorithm decomposes the raw 64-bit limb into 4 FIXED 16-bit "planes"
 * (>>16, >>32, >>48) — only valid for LIMB_BITS == 16. The other carry algorithms
 * (SINGLE_TILE/MULTI_TILE/SEQUENTIAL) are parametric in LIMB_BITS.
 */
#if LIMB_BITS != 16
#error "CARRY_ALG_PREFIX_SCAN requires LIMB_BITS == 16 (decomposition into 16-bit planes). Use another CARRY_NORM_ALG for LIMB_BITS != 16."
#endif

static constexpr int PSCAN_TILE = MR_CARRY_TILE;
static_assert(PSCAN_TILE >= 32 && PSCAN_TILE <= 1024 && (PSCAN_TILE % 32) == 0,
              "MR_CARRY_TILE must be a multiple of 32 between 32 and 1024");

static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#if PSCAN_TILE == 32
/* ───── 1-warp path (PSCAN_TILE == 32): shuffle only, no shared, no barrier ────── */

/* Carry-lookahead addition of two already-normalized digits, within a single warp.
 * `s` = A[lane] + B[lane] (≤ 2·LIMB_MASK, binary carry). `cin` is the scalar carry
 * entering the tile (identical across all lanes). Returns the lane's normalized
 * digit; writes into *tile_cout the carry leaving the tile (identical across all lanes).
 * All via __shfl — no __syncthreads and no shared memory.
 */
__device__ static inline uint64_t cla_warp(uint64_t s, int cin, int *tile_cout)
{
    int lane = threadIdx.x;
    unsigned g = (unsigned)((s >> LIMB_BITS) & 1ULL);
    unsigned p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u;

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
    unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
    unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
    if (lane == 0)
    {
        eg = 0u;
        ep = 1u;
    }
    int carry_in = (int)(eg | (ep & (unsigned)cin));

    unsigned Gt = __shfl_sync(FULL_MASK, g, 31);
    unsigned Pt = __shfl_sync(FULL_MASK, p, 31);
    *tile_cout = (int)(Gt | (Pt & (unsigned)cin));

    return ((s & LIMB_MASK) + (uint64_t)carry_in) & LIMB_MASK;
}

/* Normalizes d_src (raw 64-bit limbs, stride=src_stride) → d_dst (stride=n_dst)
 * in base 2^16. 1 warp per candidate. Safe in-place (src == dst, src_stride ==
 * n_dst): the 3-limb halo of the previous tile stays in registers (pr1/pr2/pr3),
 * and each lane only writes its own position after already having read the raw limb.
 */
template <typename T, typename TSrc = T>
__global__ static void pscan_normalize(
    T *__restrict__ d_dst,
    const TSrc *__restrict__ d_src,
    int n_dst, int src_stride, int n_batch)
{
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int lane = threadIdx.x;
    const T *src = d_src + (size_t)cand * src_stride;
    T *dst = d_dst + (size_t)cand * n_dst;

    int cin1 = 0, cin2 = 0, cin3 = 0;
    uint64_t pr1 = 0, pr2 = 0, pr3 = 0;

#ifdef MR_ADVANCED_MONITOR
    unsigned long long local_for = 0, local_dowhile = 0;
#endif

    for (int base = 0; base < n_dst; base += PSCAN_TILE)
    {
        int j = base + lane;
        uint64_t raw0 = (j < n_dst) ? limb_ld(src[j]) : 0ULL;
#ifdef MR_ADVANCED_MONITOR
        if (lane == 0)
        {
            local_for++;
            local_dowhile += 3;
        }
#endif
        uint64_t sh1 = __shfl_up_sync(FULL_MASK, raw0, 1);
        uint64_t sh2 = __shfl_up_sync(FULL_MASK, raw0, 2);
        uint64_t sh3 = __shfl_up_sync(FULL_MASK, raw0, 3);
        uint64_t raw1 = (lane >= 1) ? sh1 : pr1;
        uint64_t raw2 = (lane >= 2) ? sh2 : (lane == 1 ? pr1 : pr2);
        uint64_t raw3 = (lane >= 3) ? sh3 : (lane == 2 ? pr1 : (lane == 1 ? pr2 : pr3));

        uint64_t c0 = raw0 & LIMB_MASK;
        uint64_t c1 = (raw1 >> 16) & LIMB_MASK;
        uint64_t c2 = (raw2 >> 32) & LIMB_MASK;
        uint64_t c3 = (raw3 >> 48) & LIMB_MASK;

        int cout;
        uint64_t r1 = cla_warp(c0 + c1, cin1, &cout);
        cin1 = cout;
        uint64_t r2 = cla_warp(c2 + c3, cin2, &cout);
        cin2 = cout;
        uint64_t digit = cla_warp(r1 + r2, cin3, &cout);
        cin3 = cout;

        if (j < n_dst)
            limb_st(dst[j], digit);

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
/* ───── Block-wide path (PSCAN_TILE > 32): hierarchical warp→block scan ──────────
 * Inclusive scan within each warp via shuffle (no barrier), then combines the
 * NWARPS warp totals in shared memory (1 __syncthreads) and applies the offset.
 * Halo neighbors cross warps, so the tile's raw limbs go to shared memory.
 */

static constexpr int PSCAN_NWARPS = PSCAN_TILE / 32;

/* Carry-lookahead addition of two already-normalized digits, cooperative across the whole block.
 * wG/wP are shared memory buffers of size PSCAN_NWARPS (per-warp totals).
 */
__device__ static uint64_t cla_block(uint64_t s, int cin,
                                     unsigned *wG, unsigned *wP, int *tile_cout)
{
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    unsigned g = (unsigned)((s >> LIMB_BITS) & 1ULL);
    unsigned p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u;

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
    unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
    unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
    if (lane == 0)
    {
        eg = 0u;
        ep = 1u;
    }

    if (lane == 31)
    {
        wG[warp] = g;
        wP[warp] = p;
    }
    __syncthreads();

    unsigned Gpre = 0u, Ppre = 1u;
    unsigned Gtot = 0u, Ptot = 1u;
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

    unsigned Cg = eg | (ep & Gpre);
    unsigned Cp = ep & Ppre;
    int carry_in = (int)(Cg | (Cp & (unsigned)cin));

    *tile_cout = (int)(Gtot | (Ptot & (unsigned)cin));

    __syncthreads();
    return ((s & LIMB_MASK) + (uint64_t)carry_in) & LIMB_MASK;
}

template <typename T, typename TSrc = T>
__global__ static void pscan_normalize(
    T *__restrict__ d_dst,
    const TSrc *__restrict__ d_src,
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
    __shared__ uint64_t sprev[3];

    if (t < 3)
        sprev[t] = 0ULL;
    int cin1 = 0, cin2 = 0, cin3 = 0;

    for (int base = 0; base < n_dst; base += PSCAN_TILE)
    {
        int j = base + t;
        sraw[t] = (j < n_dst) ? limb_ld(src[j]) : 0ULL;
        __syncthreads();
        uint64_t raw0 = sraw[t];
        uint64_t raw1 = (t >= 1) ? sraw[t - 1] : sprev[0];
        uint64_t raw2 = (t >= 2) ? sraw[t - 2] : sprev[(t == 0) ? 1 : 0];
        uint64_t raw3 = (t >= 3) ? sraw[t - 3] : sprev[2 - t];

        uint64_t c0 = raw0 & LIMB_MASK;
        uint64_t c1 = (raw1 >> 16) & LIMB_MASK;
        uint64_t c2 = (raw2 >> 32) & LIMB_MASK;
        uint64_t c3 = (raw3 >> 48) & LIMB_MASK;

        int cout;
        uint64_t r1 = cla_block(c0 + c1, cin1, wG, wP, &cout);
        cin1 = cout;
        uint64_t r2 = cla_block(c2 + c3, cin2, wG, wP, &cout);
        cin2 = cout;
        uint64_t digit = cla_block(r1 + r2, cin3, wG, wP, &cout);
        cin3 = cout;

        if (j < n_dst)
            limb_st(dst[j], digit);

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

/* Contract entry points — parameters documented in carry_launch.cuh. */

namespace carry_impl
{

inline void to_limbs(LimbT *d_out, int n_out, RawT *raw, const CarryLaunch &L)
{
    pscan_normalize<LimbT, RawT><<<L.n_batch, PSCAN_TILE, 0, L.s>>>(
        d_out, raw, n_out, L.padded, L.n_batch);
}

inline void after_vadd(LimbT *d_dst, int n_dst, const CarryLaunch &L)
{
    pscan_normalize<<<L.n_batch, PSCAN_TILE, 0, L.s>>>(d_dst, d_dst, n_dst, n_dst, L.n_batch);
}

/* This algorithm cannot fold the addend into its load, so the sum has to be
 * materialized first — which does not fit a 32-bit limb (see MR_LIMB32).
 */
inline void add_raw_and_carry(LimbT *d_dst, int n_dst, RawT *raw, const CarryLaunch &L)
{
#ifdef MR_LIMB32
#error "MR_LIMB32 needs a carry algorithm that folds the addend into its load \
(MULTI_TILE, SINGLE_TILE or SEQUENTIAL); PREFIX_SCAN does not."
#endif
    constexpr int THR = MR_THR_ADD;
    unsigned bp = (unsigned)(n_dst + THR - 1) / THR;
    vadd_from_raw_batch<LimbT, RawT><<<dim3(bp, (unsigned)L.n_batch), THR, 0, L.s>>>(
        d_dst, raw, n_dst, L.padded, L.n_batch);
    after_vadd(d_dst, n_dst, L);
}

inline void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, const CarryLaunch &L)
{
    constexpr int THR = MR_THR_ADD;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)L.n_batch), THR, 0, L.s>>>(d_a, d_a, d_b, n, L.n_batch);
    after_vadd(d_a, n, L);
}
}
