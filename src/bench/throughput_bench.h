/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/throughput_bench.h
 * ROLE   throughput phase: witness checks per second
 *
 * HOW    Tests the same candidate with one witness, back to back, for
 *        timeout_sec seconds.
 *
 * NOTE   One witness rather than the full sweep keeps the work per call
 *        fixed regardless of whether the candidate is prime — a composite
 *        would be compacted away after witness 1 and silently deflate the
 *        count.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <gmp.h>
#include "config.h"
#include "bench/bench_common.h"

// Runs one GPU witness check (gpu_test_witness) on batches of `n_items`
// copies of N (MR_BATCH_SIZE by default), back-to-back, until `timeout_sec`
// has elapsed. Waits for the in-flight batch to finish before returning (GPU
// calls here are synchronous, so this is automatic).
ThroughputStats run_gpu_throughput(const mpz_t N, int timeout_sec, unsigned long witness = 2, int n_items = MR_BATCH_SIZE);

// Runs one CPU witness check (cpu_witness.h) on N repeatedly across
// `n_threads` threads (each pinned to a core) until `timeout_sec` has elapsed.
ThroughputStats run_cpu_throughput(const mpz_t N, int timeout_sec, int n_threads, unsigned long witness = 2);
