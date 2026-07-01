// mr_internals.cu — GPU kernels and helpers for Miller-Rabin exponentiation.
#include "mr_internals.cuh"
#include "helpers/time_format.h"
#include "ops/mul/multiplier.cuh"
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>

#define CU(expr) \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess) \
        throw std::runtime_error(std::string("[CUDA] " #expr ": ")+cudaGetErrorString(_e)); \
    } while(0)

using hrc = std::chrono::high_resolution_clock;

// ── Kernel: selects table[w] for each candidate given a window of WINDOW_BITS bits ──

__global__ void select_window_kernel(
        LimbT* __restrict__        d_out,
        const LimbT* __restrict__  d_table,
        const Data64* __restrict__ d_exp,
        int msb_pos, int k,
        int n_limbs, int n_total)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_total || j >= n_limbs) return;

    int w = 0;
    for (int b = 0; b < k; b++) {
        int bp = msb_pos - b;
        if (bp >= 0) {
            int li  = bp / LIMB_BITS;
            int bit = bp % LIMB_BITS;
            if ((d_exp[(size_t)t * n_limbs + li] >> bit) & 1)
                w |= (1 << (k - 1 - b));
        }
    }

    d_out[(size_t)t * n_limbs + j] =
        d_table[(size_t)w * n_total * n_limbs + (size_t)t * n_limbs + j];
}

// ── Kernel: checks whether d_r == d_ref for each candidate in d_alive ─────────────

__global__ void check_equals_kernel(
        const LimbT* __restrict__ d_r,
        const LimbT* __restrict__ d_ref,
        uint8_t* __restrict__      d_alive,
        int n_limbs, int n_total)
{
    int t = blockIdx.x;
    if (t >= n_total || d_alive[t] != 1) return;

    __shared__ int match;
    if (threadIdx.x == 0) match = 1;
    __syncthreads();

    const LimbT* rv = d_r   + (size_t)t * n_limbs;
    const LimbT* ref = d_ref + (size_t)t * n_limbs;
    for (int j = (int)threadIdx.x; j < n_limbs; j += (int)blockDim.x)
        if (rv[j] != ref[j]) atomicAnd(&match, 0);
    __syncthreads();

    if (threadIdx.x == 0 && match)
        d_alive[t] = 2;  // passed (r == ref)
}

// ── WitnessBuffers constructor/destructor ─────────────────────────────────────

WitnessBuffers::WitnessBuffers(BatchModCtx& mont, const std::vector<uint64_t>& exp_all,
               int n_total_)
    : n_total(n_total_), n(mont.n_limbs)
{
    size_t count = (size_t)n_total * n;
    size_t tb  = count * sizeof(LimbT);
    size_t eb  = count * sizeof(Data64);
    CU(cudaMalloc(&d_r,       tb));
    CU(cudaMalloc(&d_base,    tb));
    CU(cudaMalloc(&d_scratch, tb));
    CU(cudaMalloc(&d_one,     tb));
    CU(cudaMalloc(&d_cur_mul, tb));
    CU(cudaMalloc(&d_exp_dev, eb));
    CU(cudaMalloc(&d_passed,  (size_t)n_total));
    CU(cudaMemcpy(d_exp_dev, exp_all.data(), eb, cudaMemcpyHostToDevice));

    // 1 in Montgomery form
    std::vector<uint64_t> one_all(count, 0);
    for (int t = 0; t < n_total; t++) one_all[t*n] = 1;
    std::vector<uint64_t> one_mont;
    mont.to_residue_batch(one_all, one_mont);
    CU(limb_upload(d_one, one_mont.data(), count));
}

WitnessBuffers::~WitnessBuffers() {
    cudaFree(d_r);   cudaFree(d_base); cudaFree(d_scratch);
    cudaFree(d_one); cudaFree(d_cur_mul); cudaFree(d_exp_dev);
    cudaFree(d_passed);
}

// ── compact helpers ───────────────────────────────────────────────────────────

std::vector<int> compact_arrays(
        const std::vector<int>& keep,
        int n,
        std::vector<uint64_t>& N_cur,
        std::vector<uint64_t>& exp_cur,
        std::vector<uint64_t>& Nm1_cur,
        const std::vector<int>& orig_idx)
{
    int new_n = (int)keep.size();
    std::vector<uint64_t> N_new(new_n * n), exp_new(new_n * n), Nm1_new(new_n * n);
    std::vector<int> orig_new(new_n);
    for (int i = 0; i < new_n; i++) {
        int src = keep[i];
        std::copy(N_cur.begin()   + src*n, N_cur.begin()   + (src+1)*n, N_new.begin()   + i*n);
        std::copy(exp_cur.begin() + src*n, exp_cur.begin() + (src+1)*n, exp_new.begin() + i*n);
        std::copy(Nm1_cur.begin() + src*n, Nm1_cur.begin() + (src+1)*n, Nm1_new.begin() + i*n);
        orig_new[i] = orig_idx[src];
    }
    N_cur   = std::move(N_new);
    exp_cur = std::move(exp_new);
    Nm1_cur = std::move(Nm1_new);
    return orig_new;
}

// ── Sliding-window exponentiation loop ───────────────────────────────

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
        bool show_progress)
{
    int n = mont.n_limbs;
    size_t total_bytes = (size_t)n_total * n * sizeof(LimbT);

    LimbT* d_table;
    CU(cudaMalloc(&d_table, (size_t)WINDOW_SIZE * total_bytes));

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0)); CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]() { float ms=0; CU(cudaEventSynchronize(ev1)); CU(cudaEventElapsedTime(&ms,ev0,ev1)); return ms; };

    CU(cudaEventRecord(ev0));
    CU(cudaMemcpy(d_table, d_one_res_h, total_bytes, cudaMemcpyDeviceToDevice));
    CU(cudaMemcpy(d_table + (size_t)1 * n_total * n, d_base, total_bytes, cudaMemcpyDeviceToDevice));
    for (int w = 2; w < WINDOW_SIZE; w++)
        mont.modmul_batch(d_table + (size_t)(w-1)*n_total*n, d_base, d_table + (size_t)w*n_total*n);
    CU(cudaEventRecord(ev1));
    perf.table_ms += elapsed_ms();

    // Find the real MSB among all candidates
    int msb = n * LIMB_BITS - 1;
    while (msb > 0) {
        int li = msb/LIMB_BITS, bit = msb%LIMB_BITS;
        bool any = false;
        for (int t = 0; t < n_total && !any; t++)
            if ((exp_all[t*n + li] >> bit) & 1) any = true;
        if (any) break;
        msb--;
    }

    int n_windows = msb / WINDOW_BITS + 1;
    int start_win = n_windows * WINDOW_BITS - 1;

    std::vector<bool> any_nonzero(n_windows, false);
    for (int wi = 0; wi < n_windows; wi++) {
        int i = start_win - wi * WINDOW_BITS;
        for (int t = 0; t < n_total && !any_nonzero[wi]; t++)
            for (int b = 0; b < WINDOW_BITS && !any_nonzero[wi]; b++) {
                int bp = i - b;
                if (bp >= 0 && bp <= msb) {
                    if ((exp_all[t*n + bp/LIMB_BITS] >> (bp%LIMB_BITS)) & 1)
                        any_nonzero[wi] = true;
                }
            }
    }

    const int thr = MR_THR_SELECT_WIN;
    dim3 grid_sel((unsigned)(n + thr-1)/thr, (unsigned)n_total);

    auto t_start      = hrc::now();
    auto t_last_print = t_start;
    int  last_print_bits = 0;

    for (int win = 0; win < n_windows; win++) {
        int i = start_win - win * WINDOW_BITS;

        auto t_sq0 = hrc::now();
        for (int sq = 0; sq < WINDOW_BITS; sq++) {
            mont.modsq_batch(d_r, d_scratch);
            std::swap(d_r, d_scratch);
        }
        perf.sq_ms    += std::chrono::duration<float,std::milli>(hrc::now()-t_sq0).count();
        perf.sq_calls += WINDOW_BITS;

        if (any_nonzero[win]) {
            auto t_mul0 = hrc::now();
            select_window_kernel<<<grid_sel, thr>>>(d_cur_mul, d_table, d_exp_dev, i, WINDOW_BITS, n, n_total);
            mont.modmul_batch(d_r, d_cur_mul, d_scratch);
            std::swap(d_r, d_scratch);
            perf.mul_ms += std::chrono::duration<float,std::milli>(hrc::now()-t_mul0).count();
            perf.mul_calls++;
        }

        if (show_progress) {
            auto now = hrc::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now-t_last_print).count() >= MR_PROGRESS_INTERVAL_MS
                || win == n_windows-1)
            {
                int done_bits  = (win+1) * WINDOW_BITS;
                int total_bits = n_windows * WINDOW_BITS;
                double ms = std::chrono::duration_cast<std::chrono::milliseconds>(now-t_start).count();
                double dms   = std::chrono::duration<double,std::milli>(now-t_last_print).count();
                int    dbits = done_bits - last_print_bits;
                printf("\r    bit %d/%d  %3d%%  %s  %s/iter   ",
                       done_bits, total_bits, done_bits*100/total_bits,
                       fmt_time_ms(ms).c_str(),
                       fmt_time_ms(dbits>0 ? dms/dbits : 0.0).c_str());
                fflush(stdout);
                t_last_print   = now;
                last_print_bits = done_bits;
            }
        }
    }
    if (show_progress) printf("\n");

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_table);
}

// ── Prints performance report ─────────────────────────────────────────

void print_perf_simple(const PerfCtrs& perf)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms  = window_ms + perf.check_ms + perf.setup_ms + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v) { return total_ms > 0 ? v*100.0f/total_ms : 0.0f; };

    double memcpy_gb   = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0 ? memcpy_gb / (perf.memcpy_ms / 1000.0) : 0.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Time profile — WINDOW_BITS=%-2d                              ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n", fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms/perf.sq_calls : 0.0).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld win, %s/win)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms/perf.mul_calls : 0.0).c_str());
    printf("  table pre-compute       %12s  %5.1f%%\n", fmt_time_ms(perf.table_ms).c_str(), pct(perf.table_ms));
    printf("  CPU setup (to_mont)     %12s  %5.1f%%\n", fmt_time_ms(perf.setup_ms).c_str(), pct(perf.setup_ms));
    printf("  check                   %12s  %5.1f%%\n", fmt_time_ms(perf.check_ms).c_str(), pct(perf.check_ms));
    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    printf("  memcpy setup            %12s  %5.1f%%  %s\n", fmt_time_ms(perf.memcpy_ms).c_str(), pct(perf.memcpy_ms), gbps);
    printf("  ──────────────────────  %12s\n", fmt_time_ms(total_ms).c_str());
    carry_stats_print_and_reset();
}

void print_perf(const PerfCtrs& perf, BatchModCtx& mont)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms  = window_ms + perf.check_ms + perf.setup_ms + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v) { return total_ms > 0 ? v*100.0f/total_ms : 0.0f; };

    double memcpy_gb   = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0 ? memcpy_gb / (perf.memcpy_ms / 1000.0) : 0.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Time profile — WINDOW_BITS=%-2d                              ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n", fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms/perf.sq_calls : 0.0).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld win, %s/win)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms/perf.mul_calls : 0.0).c_str());

    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    std::vector<BatchModCtx::HostPhase> host = {
        {"table pre-compute", perf.table_ms, ""},
        {"CPU setup (to_mont)", perf.setup_ms, ""},
        {"check + memcpy", perf.check_ms, ""},
        {"memcpy setup", perf.memcpy_ms, gbps},
    };
    mont.print_perf(total_ms, host);
    carry_stats_print_and_reset();
}

// ── merge_perf_tree ───────────────────────────────────────────────────────────
// Merges src into dst: adds ms/calls for matching nodes (by position).
// On first call (dst.children is empty) the tree structure is cloned from src.

void merge_perf_tree(PerfNode& dst, const PerfNode& src)
{
    dst.ms    += src.ms;
    dst.calls += src.calls;
    if (src.children.empty()) return;

    if (dst.children.empty()) {
        for (auto& c : src.children) {
            dst.children.push_back(std::make_unique<PerfNode>(c->name));
            dst.children.back()->note = c->note;
            merge_perf_tree(*dst.children.back(), *c);
        }
    } else {
        for (size_t i = 0; i < src.children.size() && i < dst.children.size(); i++)
            merge_perf_tree(*dst.children[i], *src.children[i]);
    }
}

// ── print_perf_accumulated ────────────────────────────────────────────────────
// Prints the full timing report: GPU kernel tree (accumulated across all
// sub-batches/witnesses) annotated with host phases from PerfCtrs.

void print_perf_accumulated(const PerfCtrs& perf, PerfNode& tree)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms  = window_ms + perf.check_ms + perf.setup_ms
                    + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v) { return total_ms > 0 ? v*100.0f/total_ms : 0.0f; };

    double memcpy_gb   = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0
                       ? memcpy_gb / (perf.memcpy_ms / 1000.0) : 0.0;

    // ── Summary header (same as before) ──────────────────────────────────────
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Time profile — WINDOW_BITS=%-2d                              ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n",
           fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms/perf.sq_calls : 0.0f).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld win, %s/win)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms/perf.mul_calls : 0.0f).c_str());
    printf("  table pre-compute       %12s  %5.1f%%\n",
           fmt_time_ms(perf.table_ms).c_str(), pct(perf.table_ms));
    printf("  CPU setup (to_mont)     %12s  %5.1f%%\n",
           fmt_time_ms(perf.setup_ms).c_str(), pct(perf.setup_ms));
    printf("  check                   %12s  %5.1f%%\n",
           fmt_time_ms(perf.check_ms).c_str(), pct(perf.check_ms));
    {
        char gbps[32];
        snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
        printf("  memcpy setup            %12s  %5.1f%%  %s\n",
               fmt_time_ms(perf.memcpy_ms).c_str(), pct(perf.memcpy_ms), gbps);
    }
    printf("  ──────────────────────  %12s\n", fmt_time_ms(total_ms).c_str());

    // ── Inject host phases as a synthetic subtree
    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    std::vector<BatchModCtx::HostPhase> host = {
        {"table pre-compute",  perf.table_ms,  ""},
        {"CPU setup (to_mont)", perf.setup_ms, ""},
        {"check",              perf.check_ms,  ""},
        {"memcpy setup",       perf.memcpy_ms, gbps},
    };
    if (!host.empty()) {
        PerfNode* h = tree.branch("setup / host");
        for (auto& hp : host) {
            PerfNode* leaf = h->branch(hp.name);
            leaf->ms    = hp.ms;
            leaf->calls = 1;
            leaf->note  = hp.note;
        }
    }

    // "others (overhead)" = total measured by PerfCtrs minus GPU kernel tree
    if (total_ms > 0.0) {
        double gpu_tree_ms = 0.0;
        for (auto& c : tree.children) gpu_tree_ms += c->total_ms();
        double others = (double)total_ms - gpu_tree_ms;
        if (others > 0.5) {
            PerfNode* o = tree.branch("others (overhead)");
            o->ms    = others;
            o->calls = 1;
        }
    }

    printf("\n  Application breakdown (accumulated across all sub-batches):\n");
    print_perf_tree(tree);
    carry_stats_print_and_reset();
}
