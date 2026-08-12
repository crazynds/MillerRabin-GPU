/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/tests/test_known_primes.cu
 * ROLE   known Mersenne primes must come back prime
 *
 * HOW    Runs the full Miller-Rabin path on 2^p - 1 for p = 521, 607, 1279
 *        — primes small enough to finish instantly and large enough to
 *        exercise the multi-limb path.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from correctness_tests.cuh; the entry point lost
 *               its `static` and is now declared in tests/tests.cuh.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "config.h"
#include "mod/batch_mod_ctx.cuh"
#include "mr/miller_rabin_runner.cuh"
#include "tests/gmp_reference.cuh"
#include "driver/candidate.cuh"
#include <gmp.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "tests/tests.cuh"

void run_known_prime_tests()
{
    // Exponents of known Mersenne primes with > 128 bits
    static const int MERSENNE_EXP[] = {521, 607, 1279};
    static const int N_MERSENNE = (int)(sizeof(MERSENNE_EXP) / sizeof(MERSENNE_EXP[0]));

    printf("\n=== Tests with known Mersenne primes ===\n");
    printf("  M_p = 2^p - 1 for p = 521, 607, 1279\n\n");

    int max_digits = (int)(MERSENNE_EXP[N_MERSENNE - 1] * 0.30103) + 4;
    int n_limbs = limbs_for_digits(max_digits + 4);

    int nb = N_MERSENNE;
    std::vector<uint64_t> N_all((size_t)nb * n_limbs, 0);
    std::vector<uint64_t> Nm1_all((size_t)nb * n_limbs, 0);
    std::vector<uint64_t> d_all((size_t)nb * n_limbs, 0);

    mpz_t M, Mm1, d;
    mpz_inits(M, Mm1, d, nullptr);
    for (int i = 0; i < nb; i++)
    {
        mpz_ui_pow_ui(M, 2, (unsigned long)MERSENNE_EXP[i]);
        mpz_sub_ui(M, M, 1);

        mpz_sub_ui(Mm1, M, 1);
        mpz_tdiv_q_2exp(d, Mm1, 1);

        auto to_lims = [&](uint64_t *out, const mpz_t x)
        {
            mpz_t tmp;
            mpz_init_set(tmp, x);
            for (int j = 0; j < n_limbs; j++)
            {
                out[j] = mpz_get_ui(tmp) & LIMB_MASK;
                mpz_tdiv_q_2exp(tmp, tmp, LIMB_BITS);
            }
            mpz_clear(tmp);
        };

        to_lims(N_all.data() + i * n_limbs, M);
        to_lims(Nm1_all.data() + i * n_limbs, Mm1);
        to_lims(d_all.data() + i * n_limbs, d);
    }
    mpz_clears(M, Mm1, d, nullptr);

    printf("  n_limbs=%d  NTT padded=%d\n", n_limbs, next_pow2_ntt(2 * n_limbs));

    auto alive = gpu_miller_rabin_s1(N_all, d_all, Nm1_all, n_limbs, nb, DEFAULT_WITNESSES, "Mersenne");

    printf("\n  Results:\n");
    int ok = 0, fail = 0;
    for (int i = 0; i < nb; i++)
    {
        bool passed = alive[i];
        printf("  M%-4d (2^%d-1): %s\n",
               MERSENNE_EXP[i], MERSENNE_EXP[i], passed ? "PRIME OK" : "FAILED (bug!)");
        if (passed)
            ok++;
        else
            fail++;
    }

    if (fail == 0)
        printf("\n  All %d Mersenne primes correctly identified.\n", ok);
    else
        printf("\n  ERROR: %d prime(s) not identified — bug in the algorithm!\n", fail);

    printf("=== End of Mersenne tests ===\n\n");
}
