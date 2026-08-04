#pragma once
// bench_prime_finder.h — picks a random candidate for the self-benchmark
// suite: `digits` decimal digits, odd, and free of small prime factors.
//
// It is deliberately NOT required to be an actual prime — a real
// prime-hunting pipeline already filters out numbers with small factors
// (cheap) before ever invoking the expensive Miller-Rabin test, and that is
// exactly the workload this candidate represents. Requiring genuine
// primality would make candidate generation itself pay for a full
// Miller-Rabin run (and, at high digit counts, an increasingly unlikely
// one — prime density drops as 1/ln(N)) which defeats the point of a setup
// step that should be near-instant regardless of `digits`.

#include <gmp.h>

// Fills `out` (caller-owned, must already be mpz_init'd) with a random
// `digits`-digit odd number that has no factor among the primes below
// `trial_limit`. 2^32 (all 32-bit primes, ~203M of them) is impractical to
// sieve for a one-off setup step, so this defaults to 15485863 — the
// 1,000,000th prime — giving exactly the first 1M primes to trial-divide
// against.
void find_bench_candidate(mpz_t out, int digits, unsigned long trial_limit = 15485863);
