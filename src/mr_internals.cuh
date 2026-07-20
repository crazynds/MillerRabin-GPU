#pragma once
#include "batch_mod_ctx.cuh"
#include "config.h"
#include <vector>
#include <cstdint>

#define MR_WINDOW_SIZE (1 << MR_WINDOW_BITS)
static constexpr int WINDOW_BITS = MR_WINDOW_BITS;
static constexpr int WINDOW_SIZE = MR_WINDOW_SIZE;

// ── Performance counters structure ────────────────────────────────────

struct PerfCtrs {
    float sq_ms      = 0, mul_ms    = 0;
    float table_ms   = 0, check_ms  = 0;
    float setup_ms   = 0;   // CPU: to_mont + preparation
    float memcpy_ms  = 0;   // only the setup cudaMemcpy
    size_t memcpy_bytes = 0;
    long  sq_calls = 0, mul_calls = 0;
};

// ── Allocates working buffers for the witnesses ───────────────────────────────

struct WitnessBuffers {
    LimbT  *d_r, *d_base, *d_scratch, *d_one, *d_cur_mul;
    Data64 *d_exp_dev; // exponent: bit-addressed, never transformed → stays Data64
    uint8_t* d_passed;
    int n_total, n;

    WitnessBuffers(BatchModCtx& mont, const std::vector<uint64_t>& exp_all,
                   int n_total_);
    ~WitnessBuffers();
};

// ── Kernel declarations ───────────────────────────────────────────────────────

__global__ void select_window_kernel(
        LimbT* __restrict__        d_out,
        const LimbT* __restrict__  d_table,
        const Data64* __restrict__ d_exp,
        int msb_pos, int k,
        int n_limbs, int n_total);

__global__ void check_equals_kernel(
        const LimbT* __restrict__ d_r,
        const LimbT* __restrict__ d_ref,
        uint8_t* __restrict__      d_alive,
        int n_limbs, int n_total);

// ── Function declarations ─────────────────────────────────────────────────────

std::vector<int> compact_arrays(
        const std::vector<int>& keep,
        int n,
        std::vector<uint64_t>& N_cur,
        std::vector<uint64_t>& exp_cur,
        std::vector<uint64_t>& Nm1_cur,
        const std::vector<int>& orig_idx);

void window_exp_loop(
        BatchModCtx& mont,
        const std::vector<uint64_t>& exp_all,
        LimbT*& d_r,
        LimbT* d_one_res_h,
        LimbT* d_base,
        LimbT*& d_scratch,
        LimbT* d_cur_mul,
        Data64* d_exp_dev,
        int n_total,
        PerfCtrs& perf,
        uint32_t witness,
        bool show_progress,
        bool collect_perf = false);

void print_perf(const PerfCtrs& perf, BatchModCtx& mont);
void print_perf_simple(const PerfCtrs& perf);

// Merges src PerfNode tree into dst (same-structure trees; adds ms/calls recursively).
void merge_perf_tree(PerfNode& dst, const PerfNode& src);

// Prints the combined timing report using accumulated PerfCtrs (host phases) +
// accumulated PerfNode tree (GPU kernel breakdown). Call once per round at the end.
void print_perf_accumulated(const PerfCtrs& perf, PerfNode& tree);
