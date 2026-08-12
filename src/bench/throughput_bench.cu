/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/throughput_bench.cu
 * ROLE   throughput phase of the self-benchmark
 *
 * HOW    Runs single-witness checks back to back until the timeout expires,
 *        counting completions. The GPU side fills a full batch; the CPU
 *        side is swept across thread counts by the caller.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "bench/throughput_bench.h"
#include "bench/cpu_witness.h"
#include "driver/candidate.cuh"
#include "mr/miller_rabin_runner.cuh"
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <pthread.h>
#include <sched.h>

using hrc = std::chrono::high_resolution_clock;

static double seconds_since(hrc::time_point t0)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(hrc::now() - t0).count() / 1e6;
}

// ── GPU throughput ────────────────────────────────────────────────────────────

ThroughputStats run_gpu_throughput(const mpz_t N, int timeout_sec, unsigned long witness, int n_items)
{
    int digits = (int)mpz_sizeinbase(N, 10) + 4;
    int n_limbs = limbs_for_digits(digits);

    NumberCandidate cand;
    cand.build_from_mpz(N, n_limbs);

    int n_total = n_items;
    std::vector<uint64_t> N_all((size_t)n_total * n_limbs);
    std::vector<uint64_t> d_all((size_t)n_total * n_limbs);
    std::vector<uint64_t> Nm1_all((size_t)n_total * n_limbs);
    for (int t = 0; t < n_total; t++) {
        std::copy(cand.N_lims.begin(),   cand.N_lims.end(),   N_all.begin()   + (size_t)t * n_limbs);
        std::copy(cand.d_lims.begin(),   cand.d_lims.end(),   d_all.begin()   + (size_t)t * n_limbs);
        std::copy(cand.Nm1_lims.begin(), cand.Nm1_lims.end(), Nm1_all.begin() + (size_t)t * n_limbs);
    }

    ThroughputStats stats;
    auto t0 = hrc::now();
    do {
        std::vector<uint64_t> N_run = N_all, d_run = d_all, Nm1_run = Nm1_all;
        gpu_test_witness(N_run, d_run, Nm1_run, n_limbs, n_total, cand.s,
                          (uint32_t)witness, nullptr, nullptr, false);
        stats.count += n_total;
        stats.elapsed_s = seconds_since(t0);
    } while (stats.elapsed_s < timeout_sec);

    return stats;
}

// ── CPU throughput ────────────────────────────────────────────────────────────

ThroughputStats run_cpu_throughput(const mpz_t N, int timeout_sec, int n_threads, unsigned long witness)
{
    if (n_threads < 1) n_threads = 1;

    std::atomic<bool> stop{false};
    std::vector<long long> local_count(n_threads, 0);
    std::vector<double> local_elapsed_s(n_threads, 0.0);

    auto worker = [&](int idx) {
        CpuWitnessCtx ctx;
        cpu_witness_ctx_init(ctx, N);
        auto t_local = hrc::now();
        long long count = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            cpu_test_witness(ctx, witness);
            count++;
        }
        local_count[idx] = count;
        local_elapsed_s[idx] = seconds_since(t_local);
        cpu_witness_ctx_clear(ctx);
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    int n_cores = (int)std::thread::hardware_concurrency();
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back(worker, t);
        if (n_cores > 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(t % n_cores, &cpuset);
            pthread_setaffinity_np(threads.back().native_handle(), sizeof(cpu_set_t), &cpuset);
        }
    }

    std::this_thread::sleep_for(std::chrono::seconds(timeout_sec));
    stop.store(true, std::memory_order_relaxed);
    for (auto &th : threads) th.join();

    long long total_count = 0;
    double total_per_sec = 0.0;
    for (int t = 0; t < n_threads; t++) {
        total_count += local_count[t];
        if (local_elapsed_s[t] > 0.0)
            total_per_sec += local_count[t] / local_elapsed_s[t];
    }

    ThroughputStats stats;
    stats.count = total_count;
    stats.elapsed_s = total_per_sec > 0.0 ? total_count / total_per_sec : 0.0;
    return stats;
}
