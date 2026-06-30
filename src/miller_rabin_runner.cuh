#pragma once
// miller_rabin_runner.cuh — GPU execution of the Miller-Rabin test for a batch.
//
// gpu_miller_rabin_s1: optimized for N-1 = 2*d (s=1, N ≡ 3 mod 4).
// gpu_miller_rabin:    general version for N-1 = 2^s * d, any s >= 1.

#include "batch_mod_ctx.cuh"
#include "mr_internals.cuh"
#include <vector>

inline const std::vector<uint32_t> DEFAULT_WITNESSES = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

// Selects table[w] for each candidate given a window of WINDOW_BITS bits.
// Declared here so it can be referenced in correctness_tests.cuh.
__global__ void select_window_kernel(
    LimbT *__restrict__,        // d_out (residue table entry)
    const LimbT *__restrict__,  // d_table (residues)
    const Data64 *__restrict__, // d_exp (exponent: bit-addressed, stays Data64)
    int, int, int, int);

// For numbers where N-1 = 2*d (s=1).
// N_all: flat [n_total * n_limbs] moduli. exp_all: d = (N-1)/2.
// Compacts dead candidates between witnesses so the GPU batch shrinks.
std::vector<bool> gpu_miller_rabin_s1(
    const std::vector<uint64_t> &N_all,
    const std::vector<uint64_t> &exp_all,
    const std::vector<uint64_t> &Nm1_all,
    int n_limbs,
    int n_total,
    const std::vector<uint32_t> &witnesses,
    const char *label,
    bool show_report = false,
    bool show_progress = false);

// Tests one sub-batch of sub_bsz candidates with a single witness.
// Returns a passed[] vector of size sub_bsz.
// Used by the lazy-build driver in bench_mr_gpu to avoid building all candidates upfront.
std::vector<bool> gpu_test_witness(
    std::vector<uint64_t> &N_sub,
    std::vector<uint64_t> &exp_sub,
    std::vector<uint64_t> &Nm1_sub,
    int n_limbs,
    int sub_bsz,
    int s,
    uint32_t witness,
    bool show_progress = false);

// General version: N-1 = 2^s * d.
// Compacts dead candidates between witnesses so the GPU batch shrinks.
std::vector<bool> gpu_miller_rabin(
    const std::vector<uint64_t> &N_all,
    const std::vector<uint64_t> &exp_all,
    const std::vector<uint64_t> &Nm1_all,
    int s,
    int n_limbs,
    int n_total,
    const std::vector<uint32_t> &witnesses,
    const char *label,
    bool show_report = false,
    bool show_progress = false);
