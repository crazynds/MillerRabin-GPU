/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/tests/gmp_reference.cuh
 * ROLE   GMP reference arithmetic the suites compare the GPU against
 *
 * HOW    Small conversions between limb arrays and mpz_t plus the two
 *        reference operations (modular square and multiply). Grouped in one
 *        header because every suite needs all of them and none is large.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from correctness_tests.cuh.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/mul/multiplier.cuh"
#include <gmp.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "util/cuda_check.cuh"


// ── GMP helper functions ────────────────────────────────────────────────────

static void lims_to_gmp(mpz_t out, const uint64_t *lims, int nn)
{
    mpz_set_ui(out, 0);
    for (int j = nn - 1; j >= 0; j--)
    {
        mpz_mul_2exp(out, out, LIMB_BITS);
        mpz_add_ui(out, out, (unsigned long)lims[j]);
    }
}

static void gmp_to_lims(uint64_t *lims, int nn, mpz_t src)
{
    mpz_t t;
    mpz_init_set(t, src);
    for (int j = 0; j < nn; j++)
    {
        lims[j] = mpz_get_ui(t) & LIMB_MASK;
        mpz_tdiv_q_2exp(t, t, LIMB_BITS);
    }
    mpz_clear(t);
}

static bool limbs_eq(const uint64_t *a, const uint64_t *b, int n)
{
    for (int i = 0; i < n; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

static void gmp_sq_mod(uint64_t *out, const uint64_t *x, const uint64_t *N, int n)
{
    mpz_t xm, Nm, res;
    mpz_init(xm);
    mpz_init(Nm);
    mpz_init(res);
    lims_to_gmp(xm, x, n);
    lims_to_gmp(Nm, N, n);
    mpz_mul(res, xm, xm);
    mpz_mod(res, res, Nm);
    gmp_to_lims(out, n, res);
    mpz_clear(xm);
    mpz_clear(Nm);
    mpz_clear(res);
}

static void gmp_mul_mod(uint64_t *out, const uint64_t *x, const uint64_t *y,
                        const uint64_t *N, int n)
{
    mpz_t xm, ym, Nm, res;
    mpz_init(xm);
    mpz_init(ym);
    mpz_init(Nm);
    mpz_init(res);
    lims_to_gmp(xm, x, n);
    lims_to_gmp(ym, y, n);
    lims_to_gmp(Nm, N, n);
    mpz_mul(res, xm, ym);
    mpz_mod(res, res, Nm);
    gmp_to_lims(out, n, res);
    mpz_clear(xm);
    mpz_clear(ym);
    mpz_clear(Nm);
    mpz_clear(res);
}

// ── Correctness tests ───────────────────────────────────────────────────────
