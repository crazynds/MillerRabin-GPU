#pragma once
// miller_rabin_runner.cuh — GPU execution of the Miller-Rabin test for a batch.
//
// gpu_miller_rabin_s1: optimized for N-1 = 2*d (s=1, N ≡ 3 mod 4).
// gpu_miller_rabin:    general version for N-1 = 2^s * d, any s >= 1.

#include "batch_mod_ctx.cuh"
#include "config.h"
#include <vector>
#define MR_WINDOW_SIZE (1 << MR_WINDOW_BITS)

static constexpr int WINDOW_BITS = MR_WINDOW_BITS;
static constexpr int WINDOW_SIZE = MR_WINDOW_SIZE;

inline const std::vector<uint32_t> DEFAULT_WITNESSES = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

// Selects table[w] for each candidate given a window of WINDOW_BITS bits.
// Declared here so it can be referenced in correctness_tests.cuh.
__global__ void select_window_kernel(
    Data64 *__restrict__,
    const Data64 *__restrict__,
    const Data64 *__restrict__,
    int, int, int, int);

// For numbers where N-1 = 2*d (s=1).
// exp_all: d = (N-1)/2, flat [n_total * n_limbs].
std::vector<bool> gpu_miller_rabin_s1(
    BatchModCtx &mont,
    const std::vector<uint64_t> &exp_all,
    const std::vector<uint64_t> &Nm1_all,
    int n_total,
    const std::vector<uint32_t> &witnesses,
    const char *label,
    bool show_report = false,
    bool show_progress = false);

// General version: N-1 = 2^s * d.
// exp_all: d (odd), s: number of factors of 2 in N-1.
std::vector<bool> gpu_miller_rabin(
    BatchModCtx &mont,
    const std::vector<uint64_t> &exp_all,
    const std::vector<uint64_t> &Nm1_all,
    int s,
    int n_total,
    const std::vector<uint32_t> &witnesses,
    const char *label,
    bool show_report = false,
    bool show_progress = false);
