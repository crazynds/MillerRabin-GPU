#pragma once
// single_bench.h — "single candidate, single core/batch" latency phase.
//
// Unlike throughput_bench (which saturates the GPU/CPU with many copies at
// once), this measures the wall-clock cost of one witness check on exactly
// ONE candidate: GPU with a batch of 1, CPU on a single thread. This is the
// number that matters when a prime hunter has just one fresh candidate and
// wants an answer as fast as possible, rather than how many it can chew
// through in bulk.

#include <gmp.h>
#include "bench_common.h"

// Runs one GPU witness check on a single copy of N, `iters` times, and
// reports min/avg/max latency.
LatencyStats run_gpu_single(const mpz_t N, int iters, unsigned long witness = 2);

// Runs one CPU witness check (cpu_witness.h) on a single copy of N, on one
// thread, `iters` times, and reports min/avg/max latency.
LatencyStats run_cpu_single(const mpz_t N, int iters, unsigned long witness = 2);
