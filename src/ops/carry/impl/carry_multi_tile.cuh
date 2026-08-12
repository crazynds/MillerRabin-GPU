/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/impl/carry_multi_tile.cuh
 * ROLE   CARRY_ALG_MULTI_TILE — normalization in three kernels
 *
 * HOW    Phase 1 (carry_intra_copy): n_tiles x n_batch blocks each
 *        normalize MR_CARRY_TILE limbs in parallel and export the carry
 *        escaping their tile. Phase 2 (carry_propagate_tiles): injects tile
 *        t-1 escape into the head of tile t. Phase 3 (carry_inter_tiles):
 *        clears the rare residual crossing more than one tile.
 *
 * NOTE   This is the default: measured 2.8% faster than SINGLE_TILE at 100k
 *        digits, because phase 1 has n_tiles x n_batch blocks of
 *        parallelism instead of n_batch.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from carry_norm.cu; gained the carry_impl entry
 *               points.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/carry/carry_launch.cuh"
#include "ops/carry/carry_add.cuh"

static constexpr int CARRY_TILE = MR_CARRY_TILE;

/* Phase 1 — copies src -> dst and normalizes intra-tile carries in parallel.
 * Each block (tile, cand) processes CARRY_TILE elements independently; the carry
 * escaping the tile is saved in d_tile_carry[cand*n_tiles + tile].
 *
 * TSrc/TAdd are separate from T because the source is the raw INTT buffer (64-bit,
 * values up to ~2^58) while the destination holds limbs, which may be 32-bit.
 * Folding the optional addend into the initial load is what keeps the wide value in
 * a register instead of a limb slot.
 *
 * PARAMS
 *   d_dst         [out] normalized limbs, stride n_dst
 *   d_src         [in]  source coefficients, stride n_src
 *   d_tile_carry  [out] carry escaping each tile, [n_batch * n_tiles]
 *   d_first_tile  [out] first tile holding a residual, [n_batch]
 *   n_dst         limbs written per candidate
 *   n_src         stride of the source buffer
 *   n_batch       candidates in the batch
 *   d_add         [in]  optional addend folded into the load; null to disable
 *   add_stride    stride of d_add, ignored when d_add is null
 *
 * CHANGELOG
 *   2026-08-11  Split TSrc from T (MR_LIMB32) and added the fused addend d_add.
 */
template <typename T, typename TSrc = T, typename TAdd = T>
__global__ static void carry_intra_copy(
    T *__restrict__ d_dst,
    const TSrc *__restrict__ d_src,
    Data64 *__restrict__ d_tile_carry,
    int *__restrict__ d_first_tile,
    int n_dst, int n_src, int n_batch,
    const TAdd *__restrict__ d_add = nullptr, int add_stride = 0)
{
    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    if (tile == 0 && tid == 0)
        d_first_tile[cand] = n_tiles;

    int j_start = tile * CARRY_TILE;
    int j = j_start + tid;

#if MR_CARRY_TILE == 32
    uint64_t currVal = (j < n_src) ? limb_ld(d_src[cand * n_src + j]) : 0ULL;
    if (d_add && j < n_dst)
        currVal += limb_ld(d_add[cand * add_stride + j]);
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
    __shared__ uint64_t s_carry[CARRY_TILE];
    __shared__ int s_has_carry[2];
    int hc_idx = 0;

    uint64_t currVal = (j < n_src) ? limb_ld(d_src[cand * n_src + j]) : 0ULL;
    if (d_add && j < n_dst)
        currVal += limb_ld(d_add[cand * add_stride + j]);
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

/* Phase 2 — parallel single-hop inter-tile propagation (1 thread per receiver tile).
 * Thread `t` (1 <= t < n_tiles) reads the escape carry that tile t-1 produced in
 * phase 1 (d_tile_carry[t-1]), injects it into the head of its own tile, and
 * overwrites that same slot d_tile_carry[t-1] with the residual carry that escapes
 * tile t (the part that still needs to reach tile t+1, almost always 0).
 *
 * Race-free with a single buffer: slot t-1 is read and written only by thread t,
 * and tile t's limbs in d_dst are touched only by thread t. No barrier and no
 * block-size limit. The last tile's own escape (index n_tiles-1) is left
 * untouched as the overall overflow.
 *
 * PARAMS
 *   d_dst         [in,out] limbs of every tile, stride n
 *   d_tile_carry  [in,out] slot t-1 is consumed and overwritten with the residual
 *   d_first_tile  [out] earliest tile that still carries a residual
 *   n             limbs per candidate
 *   n_batch       candidates in the batch
 */
template <typename T>
__global__ static void carry_propagate_tiles(
    T *__restrict__ d_dst,
    Data64 *__restrict__ d_tile_carry,
    int *__restrict__ d_first_tile,
    int n, int n_batch)
{
    int cand = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x + 1;
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
    d_tile_carry[cand * n_tiles + (t - 1)] = (Data64)c;
    if (c != 0)
        atomicMin(&d_first_tile[cand], t + 1);
}

/* Phase 3 — sequential cleanup of the residual carries left by phase 2 (1 thread
 * per candidate). After phase 2 each tile has already absorbed the previous tile's
 * escape, so the only carries left are the residuals in d_tile_carry, almost
 * always zero; this pass is the safety net for the rare multi-tile cascade. The
 * residual escaping tile t sits in d_tile_carry[t-1] and enters tile t+1, so tile
 * m consumes slot m-2. d_first_tile[cand] holds the earliest m with a non-zero
 * residual (n_tiles if none), so most candidates exit immediately.
 *
 * PARAMS
 *   d_dst         [in,out] limbs of every tile, stride n
 *   d_tile_carry  [in]  residuals left by phase 2
 *   d_first_tile  [in]  where to start; n_tiles means nothing to do
 *   n             limbs per candidate
 *   n_batch       candidates in the batch
 */
template <typename T>
__global__ static void carry_inter_tiles(
    T *__restrict__ d_dst,
    Data64 *__restrict__ d_tile_carry,
    const int *__restrict__ d_first_tile,
    int n, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    int m_start = d_first_tile[cand];
    if (m_start >= n_tiles)
        return;

    uint64_t r = 0;
    for (int m = m_start; m < n_tiles; m++)
    {
        uint64_t c = r + (uint64_t)d_tile_carry[cand * n_tiles + (m - 2)];
        r = 0;
        if (c == 0)
            continue;
        int j_start = m * CARRY_TILE;
        int j_end = min(j_start + CARRY_TILE, n);
        for (int j = j_start; c > 0 && j < j_end; j++)
        {
            uint64_t v = limb_ld(d_dst[cand * n + j]) + c;
            limb_st(d_dst[cand * n + j], v & LIMB_MASK);
            c = v >> LIMB_BITS;
        }
        r = c;
    }
}


/* Contract entry points — parameters documented in carry_launch.cuh. */

namespace carry_impl
{

inline void to_limbs(LimbT *d_out, int n_out, RawT *raw, const CarryLaunch &L)
{
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_out + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (L.n_batch + THR - 1) / THR;
    carry_intra_copy<LimbT, RawT><<<dim3(n_tiles, L.n_batch), CARRY_TILE, 0, L.s>>>(
        d_out, raw, L.d_tile_carry, L.d_first_tile, n_out, L.padded, L.n_batch);
    if (n_tiles > 1)
    {
        carry_propagate_tiles<<<dim3((n_tiles - 1 + THR - 1) / THR, L.n_batch), THR, 0, L.s>>>(
            d_out, L.d_tile_carry, L.d_first_tile, n_out, L.n_batch);
        carry_inter_tiles<<<inter_blk, THR, 0, L.s>>>(
            d_out, L.d_tile_carry, L.d_first_tile, n_out, L.n_batch);
    }
}

inline void after_vadd(LimbT *d_dst, int n_dst, const CarryLaunch &L)
{
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (L.n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, L.n_batch), CARRY_TILE, 0, L.s>>>(
        d_dst, d_dst, L.d_tile_carry, L.d_first_tile, n_dst, n_dst, L.n_batch);
    if (n_tiles > 1)
    {
        carry_propagate_tiles<<<dim3((n_tiles - 1 + THR - 1) / THR, L.n_batch), THR, 0, L.s>>>(
            d_dst, L.d_tile_carry, L.d_first_tile, n_dst, L.n_batch);
        carry_inter_tiles<<<inter_blk, THR, 0, L.s>>>(
            d_dst, L.d_tile_carry, L.d_first_tile, n_dst, L.n_batch);
    }
}

inline void add_raw_and_carry(LimbT *d_dst, int n_dst, RawT *raw, const CarryLaunch &L)
{
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (L.n_batch + THR - 1) / THR;
    carry_intra_copy<LimbT, RawT, LimbT><<<dim3(n_tiles, L.n_batch), CARRY_TILE, 0, L.s>>>(
        d_dst, raw, L.d_tile_carry, L.d_first_tile, n_dst, L.padded, L.n_batch, d_dst, n_dst);
    if (n_tiles > 1)
    {
        carry_propagate_tiles<<<dim3((n_tiles - 1 + THR - 1) / THR, L.n_batch), THR, 0, L.s>>>(
            d_dst, L.d_tile_carry, L.d_first_tile, n_dst, L.n_batch);
        carry_inter_tiles<<<inter_blk, THR, 0, L.s>>>(
            d_dst, L.d_tile_carry, L.d_first_tile, n_dst, L.n_batch);
    }
}

inline void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, const CarryLaunch &L)
{
    constexpr int THR = MR_THR_ADD;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, L.n_batch), THR, 0, L.s>>>(d_a, d_a, d_b, n, L.n_batch);
    after_vadd(d_a, n, L);
}
}
