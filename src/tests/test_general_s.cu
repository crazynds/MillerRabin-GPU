/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/tests/test_general_s.cu
 * ROLE   primes with s != 1 exercise the general MR path
 *
 * HOW    Generates primes whose N-1 has more than one factor of two, so the
 *        runner takes the general branch with its extra squarings instead
 *        of the s == 1 fast path.
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

/* Starting points are chosen so their residues mod 8 favour the target s, which
 * makes the search converge quickly — s is still verified rigorously on the
 * prime that is found:
 *
 *   p = 5 mod 8   ->  p-1 = 4 mod 8   ->  s = 2 exactly
 *   p = 9 mod 16  ->  p-1 = 8 mod 16  ->  s = 3 exactly
 *   p = 1 mod 16  ->  p-1 = 0 mod 16  ->  s >= 4
 */
void run_general_s_prime_tests()
{
    printf("\n=== Tests with primes of s != 1 ===\n");
    printf("  Generating primes with s=2, s=3, s>=4 via GMP (> 512 bits)...\n\n");

    struct Target
    {
        unsigned long start_offset;
        int want_s_min, want_s_max;
        const char *desc;
    };
    static const Target TARGETS[] = {
        {4UL, 2, 2, "s=2"},
        {7UL, 3, 3, "s=3"},
        {15UL, 4, 99, "s>=4"},
    };
    static const int N_TARGETS = (int)(sizeof(TARGETS) / sizeof(TARGETS[0]));

    int n_limbs = limbs_for_digits(160 + 4);

    printf("  n_limbs=%d  NTT padded=%d\n\n", n_limbs, next_pow2_ntt(2 * n_limbs));

    mpz_t base, p, Nm1, d_tmp;
    mpz_inits(base, p, Nm1, d_tmp, nullptr);
    mpz_ui_pow_ui(base, 2, 512);

    int total_ok = 0, total_fail = 0;

    for (int ti = 0; ti < N_TARGETS; ti++)
    {
        const Target &tgt = TARGETS[ti];

        mpz_add_ui(p, base, tgt.start_offset);
        if (mpz_even_p(p))
            mpz_add_ui(p, p, 1);

        int found_s = -1;
        int attempts = 0;
        while (found_s < tgt.want_s_min || found_s > tgt.want_s_max)
        {
            mpz_nextprime(p, p);
            mpz_sub_ui(Nm1, p, 1);
            mpz_set(d_tmp, Nm1);
            found_s = 0;
            while (mpz_even_p(d_tmp))
            {
                mpz_tdiv_q_2exp(d_tmp, d_tmp, 1);
                found_s++;
            }
            attempts++;
            if (attempts > 2000)
            {
                found_s = -1;
                break;
            }
        }

        if (found_s == -1)
        {
            printf("  [%s] SKIP — not found in 2000 attempts\n", tgt.desc);
            continue;
        }

        if (!mpz_probab_prime_p(p, 25))
        {
            printf("  [%s] internal ERROR: found number is not prime according to GMP!\n", tgt.desc);
            total_fail++;
            continue;
        }

        char *p_str = mpz_get_str(nullptr, 10, p);
        int p_digits = (int)strlen(p_str);
        free(p_str);
        printf("  [%s] prime with %d digits found in %d attempts (s=%d)\n",
               tgt.desc, p_digits, attempts, found_s);

        NumberCandidate cand;
        cand.build_from_mpz(p, n_limbs);

        auto alive = gpu_miller_rabin(cand.N_lims, cand.d_lims, cand.Nm1_lims,
                                      cand.s, n_limbs, 1, DEFAULT_WITNESSES, tgt.desc);

        bool passed = alive[0];
        printf("  [%s] result: %s\n\n", tgt.desc, passed ? "PRIME OK" : "FAILED (bug!)");
        if (passed)
            total_ok++;
        else
            total_fail++;
    }

    mpz_clears(base, p, Nm1, d_tmp, nullptr);

    if (total_fail == 0)
        printf("  All %d primes (s!=1) correctly identified.\n", total_ok);
    else
        printf("  ERROR: %d prime(s) not identified — bug in the algorithm!\n", total_fail);

    printf("=== End of s!=1 tests ===\n\n");
}
