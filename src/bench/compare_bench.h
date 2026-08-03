#pragma once
// compare_bench.h — top-level driver for the GPU-vs-CPU self-benchmark.
//
// Orchestrates, in order:
//   1. Find a `digits`-digit small-factor-free candidate (bench_prime_finder.h)
//      — the shared workload for every phase below.
//   2. Phase 1 — throughput: GPU once, then CPU swept across every thread
//      count from 1 up to the machine's hardware concurrency
//      (throughput_bench.h).
//   3. Phase 2 — single-candidate latency: GPU then CPU, single-threaded
//      (single_bench.h).
//   4. Prints the combined final report.
//
// This file only sequences calls and prints; it owns no algorithm of its own.

struct CompareBenchOptions {
    int digits = 100000;       // decimal digits of the test candidate (the "N" the user passes)
    int throughput_timeout_s = 30;
    int single_iters = 10;
};

void run_compare_bench(const CompareBenchOptions &opts);
