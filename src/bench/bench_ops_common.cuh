/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_ops_common.cuh
 * ROLE   shared scaffolding of the primitive benchmarks
 *
 * HOW    Sizes to sweep, the random-operand generator, the BenchResult
 *        record and the ops/day formatter. Small pieces that both the CPU
 *        and the GPU side need, so they live together instead of being
 *        duplicated.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_ops.cu (661 lines).
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "config.h"
#include "ops/mul/multiplier.cuh"
#include "util/gmp_helpers.cuh"
#include <gmp.h>
#include <string>
#include <vector>
#include <chrono>

using hrc = std::chrono::high_resolution_clock;
using dsec = std::chrono::duration<double>;

// ── Parameters ────────────────────────────────────────────────────────────────

static constexpr double BENCH_SECS = 3.0;
static constexpr int N_BATCH = MR_BATCH_SIZE;
static constexpr int BIT_SIZES_SHORT[] = {512, 1024, 2048, 4096, 8192, 16384};
static constexpr int BIT_SIZES_LONG[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

// ── Helpers ───────────────────────────────────────────────────────────────────

/* One shared generator for every benchmark row, across all three translation
 * units. A `static` here would give each .cu its own copy and only the one that
 * runs gmp_randinit would be usable — the others would fault on first draw.
 * A function-local static also removes any initialization-order question.
 */
inline gmp_randstate_t &bench_rng()
{
    static gmp_randstate_t st;
    static bool ready = false;
    if (!ready)
    {
        gmp_randinit_mt(st);
        gmp_randseed_ui(st, 0xDEADBEEF);
        ready = true;
    }
    return st;
}

static void rand_odd_mpz(mpz_t out, int n_bits, gmp_randstate_t state)
{
    mpz_urandomb(out, state, n_bits);
    mpz_setbit(out, n_bits - 1);
    mpz_setbit(out, 0);
}

struct BenchResult
{
    double ops_per_sec;
    long long n_ops;
    double elapsed_sec;
    bool skipped = false;
};

// Formats ops/day (ops_per_sec * 86400) with appropriate units (T/G/M/k).
static std::string fmt_ops_per_day(double ops_per_sec)
{
    if (ops_per_sec < 0)
        return "  ERROR ";
    double per_day = ops_per_sec * 86400.0;
    char buf[32];
    if (per_day >= 1e12)
        snprintf(buf, sizeof(buf), "%7.2f T", per_day / 1e12);
    else if (per_day >= 1e9)
        snprintf(buf, sizeof(buf), "%7.2f G", per_day / 1e9);
    else if (per_day >= 1e6)
        snprintf(buf, sizeof(buf), "%7.2f M", per_day / 1e6);
    else if (per_day >= 1e3)
        snprintf(buf, sizeof(buf), "%7.2f k", per_day / 1e3);
    else if (per_day >= 1.0)
        snprintf(buf, sizeof(buf), "%7.2f  ", per_day);
    else
        snprintf(buf, sizeof(buf), "%7.4f  ", per_day);
    return buf;
}

static void make_nums(std::vector<__mpz_struct> &storage, std::vector<mpz_t *> &ptrs,
                      int n_bits)
{
    storage.resize(N_BATCH);
    ptrs.resize(N_BATCH);
    for (int i = 0; i < N_BATCH; i++)
    {
        mpz_init(&storage[i]);
        rand_odd_mpz(&storage[i], n_bits, bench_rng());
        ptrs[i] = reinterpret_cast<mpz_t *>(&storage[i]);
    }
}

static void free_nums(std::vector<__mpz_struct> &storage)
{
    for (auto &m : storage)
        mpz_clear(&m);
}

/* The five fixed witnesses, identical on both sides so the comparison is fair. */
static const uint32_t MR_WIT[] = {2, 3, 5, 7, 11};
static const int N_MR_WIT = 5;

/* __mpz_struct* -> limb array (little-endian, base 2^LIMB_BITS). */
static inline void mpz_to_limbs16(uint64_t *out, int n, __mpz_struct *x)
{
    mpz_to_limbs(out, n, (const __mpz_struct *)x);
}

/* One row of the report: measured by the GMP side. */
BenchResult bench_gmp_mul_only(int n_bits, bool is_last);
BenchResult bench_gmp_sq_only(int n_bits, bool is_last);
BenchResult bench_gmp_mul(int n_bits, bool is_last);
BenchResult bench_gmp_sq(int n_bits, bool is_last);
BenchResult bench_gmp_mr(int n_bits, bool is_last);

/* Same rows, measured on the GPU. */
BenchResult bench_gpu_mul_only(int n_bits, bool is_last);
BenchResult bench_gpu_sq_only(int n_bits, bool is_last);
BenchResult bench_gpu_mul(int n_bits, bool is_last);
BenchResult bench_gpu_sq(int n_bits, bool is_last);
BenchResult bench_gpu_mr(int n_bits, bool is_last);
