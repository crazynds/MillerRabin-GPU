/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/tests/test_s1_nextprime.cu
 * ROLE   s == 1 primes from mpz_nextprime
 *
 * HOW    Takes primes straight from GMP and checks the s == 1 fast path
 *        agrees, which is the branch the production hunt actually uses.
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

/* Starting points are 3 mod 4, so p-1 = 2 mod 4 and s is exactly 1 — the branch
 * the production hunt actually takes.
 */
void run_s1_nextprime_tests()
{
    printf("\n=== Tests with s=1 primes (mpz_nextprime, > 512 bits) ===\n");

    // Start from 2^512 + 3 and advance until finding 3 primes with s=1.
    static const int N_WANT = 3;
    int n_limbs = limbs_for_digits(160 + 4);

    printf("  n_limbs=%d  NTT padded=%d\n\n", n_limbs, next_pow2_ntt(2 * n_limbs));

    mpz_t base, p, Nm1, d_tmp;
    mpz_inits(base, p, Nm1, d_tmp, nullptr);
    mpz_ui_pow_ui(base, 2, 512);
    mpz_add_ui(p, base, 3);

    int total_ok = 0, total_fail = 0, found = 0;

    while (found < N_WANT)
    {
        mpz_nextprime(p, p);

        mpz_sub_ui(Nm1, p, 1);
        mpz_set(d_tmp, Nm1);
        int s = 0;
        while (mpz_even_p(d_tmp))
        {
            mpz_tdiv_q_2exp(d_tmp, d_tmp, 1);
            s++;
        }
        if (s != 1)
            continue;

        found++;

        if (!mpz_probab_prime_p(p, 25))
        {
            printf("  [s=1 #%d] internal ERROR: number is not prime according to GMP!\n", found);
            total_fail++;
            continue;
        }

        char *p_str = mpz_get_str(nullptr, 10, p);
        int p_digits = (int)strlen(p_str);
        free(p_str);
        printf("  [s=1 #%d] prime with %d digits (s=1)\n", found, p_digits);

        NumberCandidate cand;
        cand.build_from_mpz(p, n_limbs);

        auto alive = gpu_miller_rabin(cand.N_lims, cand.d_lims, cand.Nm1_lims,
                                      cand.s, n_limbs, 1, DEFAULT_WITNESSES, "s=1");

        bool passed = alive[0];
        printf("  [s=1 #%d] result: %s\n\n", found, passed ? "PRIME OK" : "FAILED (bug!)");
        if (passed)
            total_ok++;
        else
            total_fail++;
    }

    mpz_clears(base, p, Nm1, d_tmp, nullptr);

    if (total_fail == 0)
        printf("  All %d s=1 primes correctly identified.\n", total_ok);
    else
        printf("  ERROR: %d prime(s) not identified — bug in the algorithm!\n", total_fail);

    printf("=== End of s=1 tests ===\n\n");
}
