/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/compare_bench.h
 * ROLE   driver of the GPU-vs-CPU self-benchmark
 *
 * HOW    Picks one shared candidate, then runs phase 1 (throughput: GPU
 *        once, CPU swept across every thread count up to hardware
 *        concurrency) and phase 2 (single-candidate latency, both sides
 *        single-threaded), and prints the combined report.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

struct CompareBenchOptions {
    int digits = 100000;
    int throughput_timeout_s = 30;
    int single_iters = 10;
    int gpu_items = 0;
    bool gpu_only = false;
    bool cpu_only = false;
    int  cpu_threads_fixed = 0;
    bool skip_phase1 = false;
    bool skip_phase2 = false;
};

void run_compare_bench(const CompareBenchOptions &opts);
