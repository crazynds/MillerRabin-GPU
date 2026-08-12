/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_ops_gmp.cu
 * ROLE   CPU (GMP) side of the primitive benchmarks
 *
 * HOW    Times mpz_mul / mpz_mod and a full single-candidate Miller-Rabin
 *        on one core, for the same operand sizes the GPU side sweeps. This
 *        is the baseline the GPU numbers are read against.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_ops.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "bench/bench_ops_common.cuh"
#include "mod/batch_mod_ctx.cuh"
#include "mr/miller_rabin_runner.cuh"
#include "util/gmp_helpers.cuh"
#include "config.h"
#include <gmp.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>

// ── Benchmark GMP ─────────────────────────────────────────────────────────────

// GMP: only mpz_mul, no reduction.
BenchResult bench_gmp_mul_only(int n_bits, bool is_last)
{
    __mpz_struct A, B, C;
    mpz_init(&A);
    mpz_init(&B);
    mpz_init(&C);
    rand_odd_mpz(&A, n_bits, bench_rng());
    rand_odd_mpz(&B, n_bits, bench_rng());

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do
    {
        mpz_mul(&C, &A, &B);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A);
    mpz_clear(&B);
    mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// GMP: only mpz_mul(self), no reduction.
BenchResult bench_gmp_sq_only(int n_bits, bool is_last)
{
    __mpz_struct A, C;
    mpz_init(&A);
    mpz_init(&C);
    rand_odd_mpz(&A, n_bits, bench_rng());

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do
    {
        mpz_mul(&C, &A, &A);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A);
    mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// GMP: mpz_mul + mpz_mod — equivalent to the GPU's modmul_batch (mul + reduction mod N).
BenchResult bench_gmp_mul(int n_bits, bool is_last)
{
    __mpz_struct A, B, N, C;
    mpz_init(&A);
    mpz_init(&B);
    mpz_init(&N);
    mpz_init(&C);
    rand_odd_mpz(&A, n_bits, bench_rng());
    rand_odd_mpz(&B, n_bits, bench_rng());
    rand_odd_mpz(&N, n_bits, bench_rng());

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do
    {
        mpz_mul(&C, &A, &B);
        mpz_mod(&C, &C, &N);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A);
    mpz_clear(&B);
    mpz_clear(&N);
    mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// GMP: mpz_mul(self) + mpz_mod — equivalent to the GPU's modsq_batch.
BenchResult bench_gmp_sq(int n_bits, bool is_last)
{
    __mpz_struct A, N, C;
    mpz_init(&A);
    mpz_init(&N);
    mpz_init(&C);
    rand_odd_mpz(&A, n_bits, bench_rng());
    rand_odd_mpz(&N, n_bits, bench_rng());

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do
    {
        mpz_mul(&C, &A, &A);
        mpz_mod(&C, &C, &N);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A);
    mpz_clear(&N);
    mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// ── Benchmark GPU ─────────────────────────────────────────────────────────────

// mpz_t is typedef __mpz_struct[1] — cannot be used in std::vector directly.
// We use __mpz_struct as the element and reinterpret_cast to mpz_t*.


// A single Miller-Rabin test with GMP for one N.
// Returns false if composite, true if probably prime.
static bool gmp_mr_single(__mpz_struct *N, __mpz_struct *Nm1, __mpz_struct *d, int s,
                          __mpz_struct *tmp)
{
    for (int wi = 0; wi < N_MR_WIT; wi++)
    {
        mpz_set_ui(tmp, MR_WIT[wi]);
        mpz_powm(tmp, tmp, d, N);
        if (mpz_cmp_ui(tmp, 1) == 0 || mpz_cmp(tmp, Nm1) == 0)
            continue;
        bool composite = true;
        for (int r = 1; r < s; r++)
        {
            mpz_mul(tmp, tmp, tmp);
            mpz_mod(tmp, tmp, N);
            if (mpz_cmp(tmp, Nm1) == 0)
            {
                composite = false;
                break;
            }
        }
        if (composite)
            return false;
    }
    return true;
}

// GMP Miller-Rabin: N ≡ 3 mod 4 forced (s=1), d = (N-1)/2.
BenchResult bench_gmp_mr(int n_bits, bool is_last)
{
    __mpz_struct num, Nm1, d, tmp;
    mpz_init(&num);
    mpz_init(&Nm1);
    mpz_init(&d);
    mpz_init(&tmp);
    rand_odd_mpz(&num, n_bits, bench_rng());
    mpz_setbit(&num, 1);
    mpz_sub_ui(&Nm1, &num, 1);
    mpz_tdiv_q_2exp(&d, &Nm1, 1);

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do
    {
        gmp_mr_single(&num, &Nm1, &d, 1, &tmp);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&num);
    mpz_clear(&Nm1);
    mpz_clear(&d);
    mpz_clear(&tmp);
    return {ops / elapsed, ops, elapsed};
}
