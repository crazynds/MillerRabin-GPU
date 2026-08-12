/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/carry_add.cuh
 * ROLE   element-wise additions that feed carry normalization
 *
 * HOW    Two small kernels, grouped because they do the same thing: add
 *        without propagating any carry, leaving out-of-range limbs for the
 *        normalization kernel that follows.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from carry_norm.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/carry/carry_launch.cuh"

/* d_c = d_a + d_b element-wise, no carry propagation. Both operands are normalized
 * limbs, so the sum needs one extra bit and always fits.
 *
 * PARAMS
 *   d_c      [out] d_a + d_b, one limb per thread
 *   d_a      [in]  first operand, stride n
 *   d_b      [in]  second operand, stride n
 *   n        limbs per candidate
 *   n_batch  candidates in the batch
 */
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
/* Add d_raw (stride=padded, raw INTT) into d_dst (stride=n_dst) element-wise.
 * d_dst += the raw INTT buffer (stride padded), no carry propagation. Only usable
 * when the destination is wide enough to hold a raw coefficient — see MR_LIMB32,
 * which forces the fused-addend path instead.
 *
 * PARAMS
 *   d_dst    [in,out] limb array, stride n_dst; receives d_dst + d_raw
 *   d_raw    [in]  raw INTT coefficients, stride padded
 *   n_dst    limbs per candidate in d_dst
 *   padded   stride of the raw buffer
 *   n_batch  candidates in the batch
 *
 * CHANGELOG
 *   2026-08-11  Split TSrc from T so the raw source stays 64-bit.
 */
template <typename T, typename TSrc = T>
__global__ static void vadd_from_raw_batch(
    T *__restrict__ d_dst,
    const TSrc *__restrict__ d_raw,
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
