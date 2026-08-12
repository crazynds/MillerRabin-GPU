/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_ops_gpu.cu
 * ROLE   GPU side of the primitive benchmarks
 *
 * HOW    Times the batched modular primitives (product only, product plus
 *        reduction, and a full Miller-Rabin sweep) over a batch of
 *        MR_BATCH_SIZE random moduli. Every number is per batch, so
 *        dividing by the batch size gives the per-candidate cost.
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

BenchResult bench_gpu_mul_only(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try
    {
        BatchModCtx ctx(nums, 0);
        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(LimbT);
        size_t nb_out = (size_t)N_BATCH * ctx.n_sum * sizeof(LimbT);
        LimbT *d_A, *d_B, *d_out;
        cudaMalloc(&d_A, nb);
        cudaMalloc(&d_B, nb);
        cudaMalloc(&d_out, nb_out);
        cudaMemcpy(d_A, ctx.d_one_res, nb, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_B, ctx.d_Nm1_res, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            ctx.mul_no_redc_batch(d_A, d_B, d_out);
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_out);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU mul-only %d-bit] ERROR: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

BenchResult bench_gpu_sq_only(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try
    {
        BatchModCtx ctx(nums, 0);
        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(LimbT);
        size_t nb_out = (size_t)N_BATCH * ctx.n_sum * sizeof(LimbT);
        LimbT *d_A, *d_out;
        cudaMalloc(&d_A, nb);
        cudaMalloc(&d_out, nb_out);
        cudaMemcpy(d_A, ctx.d_one_res, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            ctx.sq_no_redc_batch(d_A, d_out);
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A);
        cudaFree(d_out);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU sq-only %d-bit] ERROR: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

BenchResult bench_gpu_mul(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try
    {
        BatchModCtx ctx(nums, 0);

        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(LimbT);
        LimbT *d_A, *d_B, *d_out;
        cudaMalloc(&d_A, nb);
        cudaMalloc(&d_B, nb);
        cudaMalloc(&d_out, nb);
        cudaMemcpy(d_A, ctx.d_one_res, nb, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_B, ctx.d_Nm1_res, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            ctx.modmul_batch(d_A, d_B, d_out);
            cudaDeviceSynchronize();
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_out);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU mul %d-bit] ERROR: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

BenchResult bench_gpu_sq(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try
    {
        BatchModCtx ctx(nums, 0);

        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(LimbT);
        LimbT *d_A, *d_out;
        cudaMalloc(&d_A, nb);
        cudaMalloc(&d_out, nb);
        cudaMemcpy(d_A, ctx.d_one_res, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            ctx.modsq_batch(d_A, d_out);
            cudaDeviceSynchronize();
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A);
        cudaFree(d_out);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU sq %d-bit] ERROR: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

// GPU Miller-Rabin: uses gpu_miller_rabin_s1 (s=1, N ≡ 3 mod 4).
// exp_all = d = (N-1)/2 for each candidate.
BenchResult bench_gpu_mr(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);
    for (int i = 0; i < N_BATCH; i++)
        mpz_setbit(&storage[i], 1);

    BenchResult res = {};
    try
    {
        int nl;
        {
            BatchModCtx tmp(nums, 0);
            nl = tmp.n_limbs;
        }

        // Build N_all, exp_all = d = (N-1)/2, Nm1_all = N-1
        std::vector<uint64_t> N_all((size_t)N_BATCH * nl, 0);
        std::vector<uint64_t> exp_all((size_t)N_BATCH * nl, 0);
        std::vector<uint64_t> Nm1_all((size_t)N_BATCH * nl, 0);
        for (int i = 0; i < N_BATCH; i++)
        {
            mpz_to_limbs16(N_all.data() + (size_t)i * nl, nl, ((__mpz_struct *)*nums[i]));
            mpz_t Nm1, d;
            mpz_init(Nm1);
            mpz_init(d);
            mpz_sub_ui(Nm1, *nums[i], 1);
            mpz_tdiv_q_2exp(d, Nm1, 1);
            mpz_to_limbs16(Nm1_all.data() + (size_t)i * nl, nl, ((__mpz_struct *)Nm1));
            mpz_to_limbs16(exp_all.data() + (size_t)i * nl, nl, ((__mpz_struct *)d));
            mpz_clear(Nm1);
            mpz_clear(d);
        }

        const std::vector<uint32_t> witnesses(MR_WIT, MR_WIT + N_MR_WIT);
        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            gpu_miller_rabin_s1(N_all, exp_all, Nm1_all, nl, N_BATCH, witnesses, "bench", false, false);
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU MR %d-bit] ERROR: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}
