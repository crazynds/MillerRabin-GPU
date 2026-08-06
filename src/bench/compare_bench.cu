#include "compare_bench.h"
#include "bench_prime_finder.h"
#include "throughput_bench.h"
#include "single_bench.h"
#include "bench_common.h"
#include <gmp.h>
#include <cstdio>
#include <vector>
#include <thread>

void run_compare_bench(const CompareBenchOptions &opts)
{
    int max_threads = opts.gpu_only ? 0 : (int)std::thread::hardware_concurrency();
    if (max_threads < 1 && !opts.gpu_only) max_threads = 1;

    int cpu_lo = 1, cpu_hi = max_threads;
    if (opts.cpu_threads_fixed > 0) {
        cpu_lo = cpu_hi = opts.cpu_threads_fixed;
    }

    // Phase 2 (single-candidate latency) is always single-threaded, so its
    // result is identical for every thread count — running it under
    // --do=Ncpu with N > 1 would just repeat the same 1-thread test.
    bool skip_phase2 = opts.skip_phase2 || (opts.cpu_only && opts.cpu_threads_fixed > 1);

    const char *title = opts.gpu_only ? "GPU-only" : (opts.cpu_only ? "CPU-only" : "GPU vs CPU compare");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  %s benchmark                    ║\n", title);
    printf("╚══════════════════════════════════════════════════╝\n");
    int gpu_items = opts.gpu_items > 0 ? opts.gpu_items : MR_BATCH_SIZE;

    printf("  digits              %d\n", opts.digits);
    if (!opts.cpu_only)
        printf("  gpu batch items     %d\n", gpu_items);
    printf("  throughput timeout  %ds (per run)\n", opts.throughput_timeout_s);
    printf("  single-shot iters   %d (per side)\n", opts.single_iters);
    if (!opts.gpu_only) {
        if (opts.cpu_threads_fixed > 0)
            printf("  cpu threads         %d\n", cpu_lo);
        else
            printf("  cpu threads swept   1..%d\n", max_threads);
    }
    printf("\n");

    mpz_t N;
    mpz_init(N);
    find_bench_candidate(N, opts.digits);
    printf("\n");

    // ── Phase 1: throughput ──────────────────────────────────────────────────
    ThroughputStats gpu_thr;
    std::vector<ThroughputStats> cpu_thr(cpu_hi + 1); // 1-indexed by thread count
    if (!opts.skip_phase1) {
        printf("=== Phase 1: throughput (%ds/run) ===\n", opts.throughput_timeout_s);
        if (!opts.cpu_only) {
            printf("  Running GPU throughput...\n");
            fflush(stdout);
            gpu_thr = run_gpu_throughput(N, opts.throughput_timeout_s, 2, gpu_items);
            printf("  GPU throughput: %.2f checks/hour\n", gpu_thr.per_hour());
            fflush(stdout);
        }

        if (!opts.gpu_only) {
            for (int t = cpu_lo; t <= cpu_hi; t++) {
                printf("  Running CPU throughput (%d thread(s))...\n", t);
                fflush(stdout);
                cpu_thr[t] = run_cpu_throughput(N, opts.throughput_timeout_s, t);
                printf("  CPU (%d thr) throughput: %.2f checks/hour\n", t, cpu_thr[t].per_hour());
                fflush(stdout);
            }
        }
    } else {
        printf("=== Phase 1: throughput — skipped ===\n");
    }

    // ── Phase 2: single-candidate latency ───────────────────────────────────
    LatencyStats gpu_lat;
    LatencyStats cpu_lat;
    if (!skip_phase2) {
        printf("\n=== Phase 2: single-candidate latency (%d iters/side) ===\n", opts.single_iters);
        if (!opts.cpu_only) {
            printf("  Running GPU single-candidate test...\n");
            fflush(stdout);
            gpu_lat = run_gpu_single(N, opts.single_iters);
            printf("  GPU latency: %.3f ms/check\n", gpu_lat.avg_ms());
            fflush(stdout);
        }
        if (!opts.gpu_only) {
            printf("  Running CPU single-candidate test...\n");
            fflush(stdout);
            cpu_lat = run_cpu_single(N, opts.single_iters);
            printf("  CPU latency: %.3f ms/check\n", cpu_lat.avg_ms());
            fflush(stdout);
        }
    } else {
        if (opts.skip_phase2)
            printf("\n=== Phase 2: single-candidate latency — skipped ===\n");
        else
            printf("\n=== Phase 2: single-candidate latency — skipped (always 1-thread; use --do=1cpu to run it) ===\n");
    }

    // ── Final report ─────────────────────────────────────────────────────────
    printf("\n=== Final report (%d-digit candidate) ===\n", opts.digits);
    if (!opts.skip_phase1) {
        printf(" -- Throughput: single-witness checks/hour (batched) --\n");
        if (!opts.cpu_only)
            print_throughput_line("GPU", gpu_thr);
        if (!opts.gpu_only) {
            for (int t = cpu_lo; t <= cpu_hi; t++) {
                char label[32];
                snprintf(label, sizeof(label), "CPU (%d thr)", t);
                print_throughput_line(label, cpu_thr[t]);
            }
            if (!opts.cpu_only && cpu_thr[cpu_hi].per_sec() > 0.0)
                printf("  GPU vs best CPU speedup:   %.2fx\n",
                       gpu_thr.per_sec() / cpu_thr[cpu_hi].per_sec());
        }
    }

    if (!skip_phase2) {
        printf(" -- Latency: single-witness check on one candidate --\n");
        if (!opts.cpu_only)
            print_latency_line("GPU", gpu_lat);
        if (!opts.gpu_only) {
            print_latency_line("CPU", cpu_lat);
            if (!opts.cpu_only && gpu_lat.avg_ms() > 0.0)
                printf("  CPU/GPU speedup:   %.2fx\n", cpu_lat.avg_ms() / gpu_lat.avg_ms());
        }
    }

    mpz_clear(N);
}
