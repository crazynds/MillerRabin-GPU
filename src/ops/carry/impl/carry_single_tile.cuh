/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/impl/carry_single_tile.cuh
 * ROLE   CARRY_ALG_SINGLE_TILE — one block per candidate
 *
 * HOW    A single block of MR_CARRY_TILE threads walks every tile of one
 *        candidate, propagating the carry through shared memory, or through
 *        shuffles when the tile is exactly one warp.
 *
 * NOTE   Needs no cross-block coordination, but its parallelism is capped
 *        at n_batch blocks — measured 2.8% slower than MULTI_TILE at 100k
 *        digits.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from carry_norm.cu; gained the carry_impl entry
 *               points.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/carry/carry_launch.cuh"
#include "ops/carry/carry_add.cuh"

static constexpr int CARRY_TILE = MR_CARRY_TILE;
static_assert(CARRY_TILE >= 32 && (CARRY_TILE % 32) == 0,
              "CARRY_ALG_SINGLE_TILE requires CARRY_TILE to be a multiple of 32");

#ifdef MR_ADVANCED_MONITOR
__device__ unsigned long long g_for_count = 0;
__device__ unsigned long long g_dowhile_count = 0;
#endif

/* Normalizes one candidate per block, walking its tiles in sequence and passing the
 * carry between them through shared memory (shuffles when CARRY_TILE == 32).
 *
 * PARAMS
 *   d_src       [in]  source coefficients, stride src_stride
 *   d_dst       [out] normalized limbs, stride n
 *   n           limbs written per candidate
 *   src_stride  stride of the source buffer
 *   n_batch     candidates in the batch
 *   d_add       [in]  optional addend folded into the load; null to disable
 *   add_stride  stride of d_add, ignored when d_add is null
 *
 * CHANGELOG
 *   2026-08-11  Split TSrc from T (MR_LIMB32) and added the fused addend d_add.
 */
template <typename T, typename TSrc = T, typename TAdd = T>
__global__ static void carry_Bbits(
    const TSrc *d_src,
    T *d_dst,
    int n, int src_stride, int n_batch,
    const TAdd *__restrict__ d_add = nullptr, int add_stride = 0)
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
    for (int tile = tid; tile < n; tile += CARRY_TILE)
    {
#ifdef MR_ADVANCED_MONITOR
        if (tid == 0)
            local_for++;
#endif
        uint64_t currVal = limb_ld(d_src[src_offset + tile]);
        if (d_add)
            currVal += limb_ld(d_add[cand * add_stride + tile]);
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
    __shared__ uint64_t s_carry[CARRY_TILE];
    __shared__ int s_has_carry[2];
    int hc_idx = 0;

    for (int tile = tid; tile < n; tile += CARRY_TILE)
    {
#ifdef MR_ADVANCED_MONITOR
        local_for++;
#endif
        uint64_t currVal = limb_ld(d_src[src_offset + tile]);
        if (d_add)
            currVal += limb_ld(d_add[cand * add_stride + tile]);
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


/* Contract entry points — parameters documented in carry_launch.cuh. */

namespace carry_impl
{

inline void to_limbs(LimbT *d_out, int n_out, RawT *raw, const CarryLaunch &L)
{
    carry_Bbits<LimbT, RawT><<<L.n_batch, CARRY_TILE, 0, L.s>>>(
        raw, d_out, n_out, L.padded, L.n_batch);
}

inline void after_vadd(LimbT *d_dst, int n_dst, const CarryLaunch &L)
{
    carry_Bbits<<<L.n_batch, CARRY_TILE, 0, L.s>>>(d_dst, d_dst, n_dst, n_dst, L.n_batch);
}

inline void add_raw_and_carry(LimbT *d_dst, int n_dst, RawT *raw, const CarryLaunch &L)
{
    carry_Bbits<LimbT, RawT, LimbT><<<L.n_batch, CARRY_TILE, 0, L.s>>>(
        raw, d_dst, n_dst, L.padded, L.n_batch, d_dst, n_dst);
}

inline void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, const CarryLaunch &L)
{
    constexpr int THR = MR_THR_ADD;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)L.n_batch), THR, 0, L.s>>>(d_a, d_a, d_b, n, L.n_batch);
    after_vadd(d_a, n, L);
}
}
