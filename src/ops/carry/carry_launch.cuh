/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   ops/carry/carry_launch.cuh
 * ROLE   contract shared by carry_norm.cu and the carry algorithms
 *
 * HOW    Each carry-normalization strategy (CARRY_NORM_ALG) lives in its own
 *        impl/ file and exposes the same four entry points in namespace
 *        carry_impl. carry_norm.cu includes exactly one of them and delegates,
 *        which keeps the four-way #if out of the public methods.
 *
 *        The entry points, all inline — a single TU includes exactly one
 *        implementation, so there is no linkage cost:
 *
 *          to_limbs(d_out, n_out, raw, L)
 *              d_out = normalize(raw). Reads raw with stride L.padded.
 *          after_vadd(d_dst, n_dst, L)
 *              Normalizes d_dst in place, after a sum already materialized.
 *          add_raw_and_carry(d_dst, n_dst, raw, L)
 *              d_dst = normalize(d_dst + raw), single pass. The addend rides in
 *              the normalization kernel's load, so the wide raw value never
 *              occupies a limb slot — this is what makes MR_LIMB32 viable.
 *          add_and_carry(d_a, d_b, n, L)
 *              d_a = normalize(d_a + d_b), both n-limb arrays.
 *
 * NOTE   The four signatures are identical across every algorithm, so they are
 *        documented here once instead of in each impl/ file, where four copies
 *        would drift apart.
 *
 * PARAMS
 *   d_out / d_dst / d_a  [out] destination limb array
 *   d_b                  [in]  second operand, same geometry as d_a
 *   raw                  [in]  raw INTT coefficients, stride L.padded
 *   n_out / n_dst / n    limbs written per candidate
 *   L                    launch context, see CarryLaunch below
 *
 * CHANGELOG
 *   2026-08-11  Added when carry_norm.cu (894 lines) was split per algorithm.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "config.h"
#include "ops/mul/multiplier.cuh"

/* What the algorithms need from the Multiplier: the batch geometry (n_batch
 * candidates, raw coefficients strided by padded) and the stream every launch
 * goes to.
 *
 * The inter-tile buffers exist only under CARRY_ALG_MULTI_TILE, the one
 * algorithm that splits a candidate across blocks. The others never allocate or
 * read them, so they are compiled out instead of carried around as null.
 */
struct CarryLaunch
{
    int n_batch;
    int padded;
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry;
    int *d_first_tile;
#endif
    cudaStream_t s;
};
