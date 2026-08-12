/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/carry_norm.cu
 * ROLE   entry point of carry normalization
 *
 * HOW    After a multiplication the raw INTT coefficients are not limbs:
 *        each one can reach ~2^58 while a limb is below 2^LIMB_BITS.
 *        Normalizing means pushing that excess into the following limbs
 *        until everything fits the base again.
 *
 * NOTE   The propagation is sequential by nature and there are four
 *        strategies to parallelize it. Each lives in an impl/ file and
 *        exposes the same four entry points in namespace carry_impl; this
 *        file picks one at compile time and the public methods just
 *        delegate.
 *
 * CHANGELOG
 *   2026-08-11  Split per algorithm (was one 894-line file with a four-way
 *               #if inside each of the five public methods).
 * ───────────────────────────────────────────────────────────────────────────── */
#include "config.h"
#include "ops/mul/multiplier.cuh"
#include "ops/carry/carry_launch.cuh"
#include <cstdio>

/* Algorithm selection. Exactly one impl/ enters the build. */

#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
#include "ops/carry/impl/carry_single_tile.cuh"
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
#include "ops/carry/impl/carry_multi_tile.cuh"
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
#include "ops/carry/impl/carry_sequential.cuh"
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
#include "ops/carry/impl/carry_prefix_scan.cuh"
#else
#error "CARRY_NORM_ALG must be CARRY_ALG_SINGLE_TILE, CARRY_ALG_MULTI_TILE, CARRY_ALG_SEQUENTIAL or CARRY_ALG_PREFIX_SCAN"
#endif

/* Bundles what the algorithms need from the Multiplier.
 *
 * CHANGELOG
 *   2026-08-11  Added together with the per-algorithm split.
 */
static inline CarryLaunch launch_ctx(const Multiplier &m, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    return CarryLaunch{m.n_batch, m.padded, m.d_tile_carry, m.d_first_tile, s};
#else
    return CarryLaunch{m.n_batch, m.padded, s};
#endif
}

/* Iteration statistics of the propagation loop, only under MR_ADVANCED_MONITOR.
 * Useful for choosing CARRY_NORM_ALG / MR_CARRY_TILE; costs printf otherwise.
 */
static inline void carry_stats_impl()
{
#ifdef MR_ADVANCED_MONITOR
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    unsigned long long h_for, h_dowhile;
    cudaMemcpyFromSymbol(&h_for, g_for_count, sizeof(h_for));
    cudaMemcpyFromSymbol(&h_dowhile, g_dowhile_count, sizeof(h_dowhile));
    if (h_for > 0)
        printf("[carry_Bbits] for=%llu  do-while=%llu  mean=%.3f iter/tile\n",
               h_for, h_dowhile, (double)h_dowhile / (double)h_for);
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(g_for_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_dowhile_count, &zero, sizeof(zero));
#endif
#endif
}

void carry_stats_print_and_reset() { carry_stats_impl(); }

/* Public methods. Each delegates to the selected algorithm — see the matching
 * impl/ file for how it works.
 */

/* d_out = normalize(raw INTT coefficients), n_out limbs per candidate.
 *
 * PARAMS
 *   d_out  [out] destination limb array
 *   n_out  limbs written per candidate
 *   s      CUDA stream
 */
void Multiplier::carry_to_limbs(LimbT *d_out, int n_out, cudaStream_t s)
{
    carry_impl::to_limbs(d_out, n_out, raw_coeffs(), launch_ctx(*this, s));
}

/* Normalizes d_dst in place, after a sum already materialized into d_dst.
 *
 * PARAMS
 *   d_dst  [in,out] limb array to normalize
 *   n_dst  limbs per candidate
 *   s      CUDA stream
 */
void Multiplier::carry_after_vadd(LimbT *d_dst, int n_dst, cudaStream_t s)
{
    carry_impl::after_vadd(d_dst, n_dst, launch_ctx(*this, s));
}

/* d_dst = normalize(d_dst + raw coefficients), in a single pass.
 *
 * The addend is folded into the normalization kernel's load, so the sum happens in a
 * 64-bit register and the raw coefficient (~2^58) never has to sit in a limb slot —
 * which is what makes 32-bit limbs (MR_LIMB32) viable for Montgomery.
 *
 * PARAMS
 *   d_dst  [in,out] limb array, receives normalize(d_dst + raw)
 *   n_dst  limbs per candidate
 *   s      CUDA stream
 *
 * CHANGELOG
 *   2026-08-11  Went from vadd + carry (two passes) to the fused addend.
 */
void Multiplier::add_raw_buf_and_carry(LimbT *d_dst, int n_dst, cudaStream_t s)
{
    carry_impl::add_raw_and_carry(d_dst, n_dst, raw_coeffs(), launch_ctx(*this, s));
}

/* d_a = normalize(d_a + d_b), both n-limb arrays.
 *
 * PARAMS
 *   d_a       [in,out] first operand, receives the result
 *   d_b       [in]  second operand
 *   n         limbs per candidate
 *   n_passes  unused; kept for call-site compatibility
 *   s         CUDA stream
 */
void Multiplier::add_and_carry(LimbT *d_a, const LimbT *d_b, int n, int /*n_passes*/,
                               cudaStream_t s)
{
    carry_impl::add_and_carry(d_a, d_b, n, launch_ctx(*this, s));
}

/* d_dst += raw coefficients, WITHOUT normalizing. Kept for the old two-pass path;
 * prefer add_raw_buf_and_carry, which does the same in one pass.
 *
 * PARAMS
 *   d_dst  [in,out] limb array, receives d_dst + raw (not normalized)
 *   n_dst  limbs per candidate
 *   s      CUDA stream
 */
void Multiplier::vadd_raw_buf(LimbT *d_dst, int n_dst, cudaStream_t s)
{
#if CARRY_NORM_ALG != CARRY_ALG_SEQUENTIAL
    const CarryLaunch L = launch_ctx(*this, s);
    constexpr int THR = MR_THR_ADD;
    unsigned bp = (unsigned)(n_dst + THR - 1) / THR;
    vadd_from_raw_batch<LimbT, RawT><<<dim3(bp, (unsigned)L.n_batch), THR, 0, L.s>>>(
        d_dst, raw_coeffs(), n_dst, L.padded, L.n_batch);
#else
    (void)d_dst; (void)n_dst; (void)s;
#endif
}
