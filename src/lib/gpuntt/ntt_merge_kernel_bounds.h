#pragma once
// Ranges that the fused INTT butterfly kernels are template-specialised over,
// via dispatch_const() (see helpers/const_dispatch.h).
//
// These mirror GPU-NTT's KernelConfig tables (CreateInverseNTTKernel in
// gpuntt/ntt_merge/ntt.cuh):
//
//   - the low-ring kernel only runs for n_power < 11 (INTT) / < 10 (NTT);
//   - no config uses an outer_iteration_count above 10;
//   - every non-low-ring config has shared_index 8 or 9.
//
// If upstream ever adds a config outside these ranges, dispatch_const throws on
// the very first NTT/INTT call — widen the bound here to match.

static constexpr int MAX_LOW_RING_N_POWER = 10;     // CreateInverseNTTKernel
static constexpr int MAX_FWD_LOW_RING_N_POWER = 9;  // CreateForwardNTTKernel
static constexpr int MAX_OUTER_ITERATION_COUNT = 10;
static constexpr int MIN_SHARED_INDEX = 8;
static constexpr int MAX_SHARED_INDEX = 9;
