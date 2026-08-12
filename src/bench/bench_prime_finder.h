/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_prime_finder.h
 * ROLE   random workload candidate for the self-benchmark
 *
 * HOW    Draws an odd number of `digits` decimal digits and rejects small
 *        prime factors.
 *
 * NOTE   It is deliberately not required to be prime. A real hunt filters
 *        small factors cheaply before paying for Miller-Rabin, so this is
 *        exactly the workload the benchmark should represent. Demanding a
 *        real prime would make setup pay for a full MR run — and
 *        increasingly often, since prime density falls as 1/ln(N).
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <gmp.h>

// Fills `out` (caller-owned, must already be mpz_init'd) with a random
// `digits`-digit odd number that has no factor among the primes below
// `trial_limit`. 2^32 (all 32-bit primes, ~203M of them) is impractical to
// sieve for a one-off setup step, so this defaults to 15485863 — the
// 1,000,000th prime — giving exactly the first 1M primes to trial-divide
// against.
void find_bench_candidate(mpz_t out, int digits, unsigned long trial_limit = 15485863);
