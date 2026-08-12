/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/ntt_merge_kernel_bounds.h
 * ROLE   ranges the fused kernels are template-specialised over
 *
 * HOW    Mirrors the launch tables in GPU-NTT: the low-ring kernel only
 *        runs below a given n_power, no configuration exceeds a certain
 *        outer iteration count, and every non-low-ring configuration uses
 *        shared_index 8 or 9.
 *
 * NOTE   dispatch_const throws on the first call if upstream ever adds a
 *        configuration outside these bounds, so the failure is loud and
 *        immediate rather than a silently wrong launch.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

static constexpr int MAX_LOW_RING_N_POWER = 10;
static constexpr int MAX_FWD_LOW_RING_N_POWER = 9;
static constexpr int MAX_OUTER_ITERATION_COUNT = 10;
static constexpr int MIN_SHARED_INDEX = 8;
static constexpr int MAX_SHARED_INDEX = 9;
