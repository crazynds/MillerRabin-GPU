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
    int max_threads = (int)std::thread::hardware_concurrency();
    if (max_threads < 1) max_threads = 1;

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  GPU vs CPU compare benchmark                    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  digits              %d\n", opts.digits);
    printf("  throughput timeout  %ds (per run)\n", opts.throughput_timeout_s);
    printf("  single-shot iters   %d (per side)\n", opts.single_iters);
    printf("  cpu threads swept   1..%d\n\n", max_threads);

    mpz_t N;
    mpz_init(N);
    find_bench_candidate(N, opts.digits);
    printf("\n");

    // ── Phase 1: throughput ──────────────────────────────────────────────────
    printf("=== Phase 1: throughput (%ds/run) ===\n", opts.throughput_timeout_s);
    printf("  Running GPU throughput...\n");
    fflush(stdout);
    ThroughputStats gpu_thr = run_gpu_throughput(N, opts.throughput_timeout_s);

    std::vector<ThroughputStats> cpu_thr(max_threads + 1); // 1-indexed by thread count
    for (int t = 1; t <= max_threads; t++) {
        printf("  Running CPU throughput (%d/%d thread(s))...\n", t, max_threads);
        fflush(stdout);
        cpu_thr[t] = run_cpu_throughput(N, opts.throughput_timeout_s, t);
    }

    // ── Phase 2: single-candidate latency ───────────────────────────────────
    printf("\n=== Phase 2: single-candidate latency (%d iters/side) ===\n", opts.single_iters);
    printf("  Running GPU single-candidate test...\n");
    fflush(stdout);
    LatencyStats gpu_lat = run_gpu_single(N, opts.single_iters);
    printf("  Running CPU single-candidate test...\n");
    fflush(stdout);
    LatencyStats cpu_lat = run_cpu_single(N, opts.single_iters);

    // ── Final report ─────────────────────────────────────────────────────────
    printf("\n=== Final report (%d-digit candidate) ===\n", opts.digits);
    printf(" -- Throughput: single-witness checks/sec (batched) --\n");
    print_throughput_line("GPU", gpu_thr);
    for (int t = 1; t <= max_threads; t++) {
        char label[32];
        snprintf(label, sizeof(label), "CPU (%d thr)", t);
        print_throughput_line(label, cpu_thr[t]);
    }
    if (cpu_thr[max_threads].per_sec() > 0.0)
        printf("  GPU vs best CPU speedup:   %.2fx\n",
               gpu_thr.per_sec() / cpu_thr[max_threads].per_sec());

    printf(" -- Latency: single-witness check on one candidate --\n");
    print_latency_line("GPU", gpu_lat);
    print_latency_line("CPU", cpu_lat);
    if (gpu_lat.avg_ms() > 0.0)
        printf("  CPU/GPU speedup:   %.2fx\n", cpu_lat.avg_ms() / gpu_lat.avg_ms());

    mpz_clear(N);
}
