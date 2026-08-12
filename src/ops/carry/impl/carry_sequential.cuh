/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/impl/carry_sequential.cuh
 * ROLE   CARRY_ALG_SEQUENTIAL — one thread per candidate
 *
 * HOW    One thread walks all limbs of a candidate in order, carrying in a
 *        register. No shared memory and no barrier.
 *
 * NOTE   Correctness baseline, used to rule out races in the other
 *        algorithms. Not a production path.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from carry_norm.cu. Fixed: CARRY_TILE was never
 *               defined in this branch, so building with
 *               CARRY_ALG_SEQUENTIAL failed to compile.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/carry/carry_launch.cuh"
#include "ops/carry/carry_add.cuh"

static constexpr int CARRY_TILE = MR_CARRY_TILE;

/* 1 thread per candidate — copies d_src (stride=n_src) → d_dst (stride=n_dst)
 * normalizing carries sequentially.
 *
 * PARAMS
 *   d_dst    [out] normalized limbs, stride n_dst
 *   d_src    [in]  source coefficients, stride n_src
 *   n_dst    limbs written per candidate
 *   n_src    stride of the source buffer
 *   n_batch  candidates in the batch
 */
template <typename T, typename TSrc = T>
__global__ static void carry_sequential(
    T *__restrict__ d_dst,
    const TSrc *__restrict__ d_src,
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

/* Fused version for add_raw_buf_and_carry: adds d_raw (raw INTT, stride=padded)
 * into d_dst and normalizes carries in a single sequential pass per candidate.
 *
 * PARAMS
 *   d_dst    [in,out] limbs, stride n_dst; receives normalize(d_dst + d_raw)
 *   d_raw    [in]  raw INTT coefficients, stride padded
 *   n_dst    limbs per candidate
 *   padded   stride of the raw buffer
 *   n_batch  candidates in the batch
 */
template <typename T, typename TSrc = T>
__global__ static void vadd_carry_sequential(
    T *__restrict__ d_dst,
    const TSrc *__restrict__ d_raw,
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


/* Contract entry points — parameters documented in carry_launch.cuh. */

namespace carry_impl
{

inline void to_limbs(LimbT *d_out, int n_out, RawT *raw, const CarryLaunch &L)
{
    int blk = (L.n_batch + CARRY_TILE - 1) / CARRY_TILE;
    carry_sequential<LimbT, RawT><<<blk, CARRY_TILE, 0, L.s>>>(
        d_out, raw, n_out, L.padded, L.n_batch);
}

/* The sum was already normalized inside add_raw_and_carry — nothing to do. */
inline void after_vadd(LimbT *, int, const CarryLaunch &) {}

inline void add_raw_and_carry(LimbT *d_dst, int n_dst, RawT *raw, const CarryLaunch &L)
{
    int blk = (L.n_batch + CARRY_TILE - 1) / CARRY_TILE;
    vadd_carry_sequential<LimbT, RawT><<<blk, CARRY_TILE, 0, L.s>>>(
        d_dst, raw, n_dst, L.padded, L.n_batch);
}

inline void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, const CarryLaunch &L)
{
    constexpr int THR = MR_THR_ADD;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)L.n_batch), THR, 0, L.s>>>(d_a, d_a, d_b, n, L.n_batch);
    int blk = (L.n_batch + CARRY_TILE - 1) / CARRY_TILE;
    carry_sequential<<<blk, CARRY_TILE, 0, L.s>>>(d_a, d_a, n, n, L.n_batch);
}
}
