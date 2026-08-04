#pragma once
// throughput_bench.h — "how many single-witness Miller-Rabin checks per
// second" phase.
//
// The same candidate (see bench_prime_finder.h — small-factor-free, not
// necessarily prime) is tested with one witness, back-to-back, for
// `timeout_sec` seconds. Using a single witness (rather than the full
// DEFAULT_WITNESSES sweep) keeps the amount of work per call fixed and
// comparable regardless of whether the candidate happens to be prime: a
// composite candidate would otherwise get compacted away after witness 1
// and silently deflate the count.
//
// GPU and CPU sides are independent entry points; nothing here decides which
// to run or how to present the result — that's compare_bench's job.

#include <gmp.h>
#include "config.h"
#include "bench_common.h"

// Runs one GPU witness check (gpu_test_witness) on batches of `n_items`
// copies of N (MR_BATCH_SIZE by default), back-to-back, until `timeout_sec`
// has elapsed. Waits for the in-flight batch to finish before returning (GPU
// calls here are synchronous, so this is automatic).
ThroughputStats run_gpu_throughput(const mpz_t N, int timeout_sec, unsigned long witness = 2, int n_items = MR_BATCH_SIZE);

// Runs one CPU witness check (cpu_witness.h) on N repeatedly across
// `n_threads` threads (each pinned to a core) until `timeout_sec` has elapsed.
ThroughputStats run_cpu_throughput(const mpz_t N, int timeout_sec, int n_threads, unsigned long witness = 2);
