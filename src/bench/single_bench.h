/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/single_bench.h
 * ROLE   latency phase: one candidate, one core or a batch of one
 *
 * HOW    Measures the wall-clock cost of a single witness check on exactly
 *        one candidate.
 *
 * NOTE   This is the number that matters when a hunter has one fresh
 *        candidate and wants an answer now, as opposed to throughput_bench,
 *        which measures how many it can chew through in bulk.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <gmp.h>
#include "bench/bench_common.h"

// Runs one GPU witness check on a single copy of N, `iters` times, and
// reports min/avg/max latency.
LatencyStats run_gpu_single(const mpz_t N, int iters, unsigned long witness = 2);

// Runs one CPU witness check (cpu_witness.h) on a single copy of N, on one
// thread, `iters` times, and reports min/avg/max latency.
LatencyStats run_cpu_single(const mpz_t N, int iters, unsigned long witness = 2);
