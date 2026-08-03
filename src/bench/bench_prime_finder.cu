#include "bench_prime_finder.h"
#include <vector>
#include <cstdint>
#include <cstdio>

// Sieve of Eratosthenes up to `limit` (inclusive). Returns the primes found.
static std::vector<uint32_t> sieve_primes(unsigned long limit)
{
    std::vector<bool> is_composite(limit + 1, false);
    std::vector<uint32_t> primes;
    for (unsigned long p = 2; p <= limit; p++) {
        if (is_composite[p]) continue;
        primes.push_back((uint32_t)p);
        for (unsigned long m = p * p; m <= limit; m += p)
            is_composite[m] = true;
    }
    return primes;
}

// True if `n` has no factor among `primes`.
static bool survives_trial_division(const mpz_t n, const std::vector<uint32_t> &primes)
{
    for (uint32_t p : primes) {
        if (mpz_fdiv_ui(n, p) == 0)
            return false;
    }
    return true;
}

void find_bench_candidate(mpz_t out, int digits, unsigned long trial_limit)
{
    if (digits < 1)
        digits = 1;

    std::vector<uint32_t> primes = sieve_primes(trial_limit);

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    FILE *urandom = fopen("/dev/urandom", "rb");
    unsigned long seed = 0;
    if (urandom) {
        if (fread(&seed, sizeof(seed), 1, urandom) != 1) seed = 0;
        fclose(urandom);
    }
    gmp_randseed_ui(rng, seed);

    unsigned long bits = (unsigned long)(digits * 3.321928094887362) + 1;

    mpz_t cand;
    mpz_init(cand);

    long long attempts = 0;
    do {
        mpz_urandomb(cand, rng, bits);
        mpz_setbit(cand, bits - 1); // fix the bit length
        mpz_setbit(cand, 0);        // force odd
        attempts++;
    } while (!survives_trial_division(cand, primes));

    printf("  [bench] found %d-digit small-factor-free candidate after %lld attempt(s) "
           "(trial-divided against %zu primes)\n",
           digits, attempts, primes.size());

    mpz_set(out, cand);
    mpz_clear(cand);
    gmp_randclear(rng);
}
