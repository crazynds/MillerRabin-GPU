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

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  %s benchmark                    ║\n", opts.gpu_only ? "GPU-only" : "GPU vs CPU compare");
    printf("╚══════════════════════════════════════════════════╝\n");
    int gpu_items = opts.gpu_items > 0 ? opts.gpu_items : MR_BATCH_SIZE;

    printf("  digits              %d\n", opts.digits);
    printf("  gpu batch items     %d\n", gpu_items);
    printf("  throughput timeout  %ds (per run)\n", opts.throughput_timeout_s);
    printf("  single-shot iters   %d (per side)\n", opts.single_iters);
    if (!opts.gpu_only)
        printf("  cpu threads swept   1..%d\n", max_threads);
    printf("\n");

    mpz_t N;
    mpz_init(N);
    find_bench_candidate(N, opts.digits);
    printf("\n");

    // ── Phase 1: throughput ──────────────────────────────────────────────────
    ThroughputStats gpu_thr;
    std::vector<ThroughputStats> cpu_thr(max_threads + 1); // 1-indexed by thread count
    if (!opts.skip_phase1) {
        printf("=== Phase 1: throughput (%ds/run) ===\n", opts.throughput_timeout_s);
        printf("  Running GPU throughput...\n");
        fflush(stdout);
        gpu_thr = run_gpu_throughput(N, opts.throughput_timeout_s, 2, gpu_items);

        if (!opts.gpu_only) {
            for (int t = 1; t <= max_threads; t++) {
                printf("  Running CPU throughput (%d/%d thread(s))...\n", t, max_threads);
                fflush(stdout);
                cpu_thr[t] = run_cpu_throughput(N, opts.throughput_timeout_s, t);
            }
        }
    } else {
        printf("=== Phase 1: throughput — skipped ===\n");
    }

    // ── Phase 2: single-candidate latency ───────────────────────────────────
    LatencyStats gpu_lat;
    LatencyStats cpu_lat;
    if (!opts.skip_phase2) {
        printf("\n=== Phase 2: single-candidate latency (%d iters/side) ===\n", opts.single_iters);
        printf("  Running GPU single-candidate test...\n");
        fflush(stdout);
        gpu_lat = run_gpu_single(N, opts.single_iters);
        if (!opts.gpu_only) {
            printf("  Running CPU single-candidate test...\n");
            fflush(stdout);
            cpu_lat = run_cpu_single(N, opts.single_iters);
        }
    } else {
        printf("\n=== Phase 2: single-candidate latency — skipped ===\n");
    }

    // ── Final report ─────────────────────────────────────────────────────────
    printf("\n=== Final report (%d-digit candidate) ===\n", opts.digits);
    if (!opts.skip_phase1) {
        printf(" -- Throughput: single-witness checks/sec (batched) --\n");
        print_throughput_line("GPU", gpu_thr);
        if (!opts.gpu_only) {
            for (int t = 1; t <= max_threads; t++) {
                char label[32];
                snprintf(label, sizeof(label), "CPU (%d thr)", t);
                print_throughput_line(label, cpu_thr[t]);
            }
            if (cpu_thr[max_threads].per_sec() > 0.0)
                printf("  GPU vs best CPU speedup:   %.2fx\n",
                       gpu_thr.per_sec() / cpu_thr[max_threads].per_sec());
        }
    }

    if (!opts.skip_phase2) {
        printf(" -- Latency: single-witness check on one candidate --\n");
        print_latency_line("GPU", gpu_lat);
        if (!opts.gpu_only) {
            print_latency_line("CPU", cpu_lat);
            if (gpu_lat.avg_ms() > 0.0)
                printf("  CPU/GPU speedup:   %.2fx\n", cpu_lat.avg_ms() / gpu_lat.avg_ms());
        }
    }

    mpz_clear(N);
}
