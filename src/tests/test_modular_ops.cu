/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/tests/test_modular_ops.cu
 * ROLE   GPU modular primitives vs GMP, on random operands
 *
 * HOW    Builds a batch of random moduli, runs every primitive (round trip,
 *        raw NTT square, modular square and multiply, sq == mul, (N-1)^2,
 *        commutativity, identity, modpow, windowed exponentiation) and
 *        compares each against GMP. This is the suite that catches an
 *        arithmetic regression.
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

static void run_ops_against_gmp(BatchModCtx &mont,
                                  const std::vector<uint64_t> &N_all)
{
    int n = mont.n_limbs, nb = mont.n_batch;
    size_t total_bytes = (size_t)nb * n * sizeof(LimbT);

    printf("\n=== Correctness tests (LIMB_BITS=%d, n_batch=%d) ===\n",
           LIMB_BITS, nb);

    std::vector<uint64_t> N_h((size_t)nb * n);
    CU(limb_download(N_h.data(), mont.d_N, N_h.size()));


    auto make_large = [&](std::vector<uint64_t> &v, int offset)
    {
        v.assign((size_t)nb * n, 0);
        for (int i = 0; i < nb; i++)
        {
            const uint64_t *Ni = N_h.data() + i * n;
            uint64_t *dst = v.data() + i * n;
            std::copy(Ni, Ni + n, dst);
            uint64_t borrow = (uint64_t)(offset + i + 1);
            for (int j = 0; j < n && borrow > 0; j++)
            {
                if (dst[j] >= borrow)
                {
                    dst[j] -= borrow;
                    borrow = 0;
                }
                else
                {
                    dst[j] = dst[j] + ((1ULL << LIMB_BITS) - borrow);
                    borrow = 1;
                }
            }
        }
    };

    auto make_rand = [&](std::vector<uint64_t> &v, uint64_t seed)
    {
        v.assign((size_t)nb * n, 0);
        for (int i = 0; i < nb; i++)
        {
            mpz_t Nm, rval;
            mpz_init(Nm);
            mpz_init(rval);
            lims_to_gmp(Nm, N_h.data() + i * n, n);
            uint64_t state = seed ^ (uint64_t)(i + 1) * 6364136223846793005ULL;
            if (state == 0)
                state = 1;
            for (int j = 0; j < n; j++)
            {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                v[i * n + j] = state & LIMB_MASK;
            }
            lims_to_gmp(rval, v.data() + i * n, n);
            mpz_mod(rval, rval, Nm);
            gmp_to_lims(v.data() + i * n, n, rval);
            mpz_clear(Nm);
            mpz_clear(rval);
        }
    };

    std::vector<uint64_t> x_small((size_t)nb * n, 0);
    std::vector<uint64_t> y_small((size_t)nb * n, 0);
    for (int i = 0; i < nb; i++)
    {
        x_small[i * n] = (uint64_t)(i + 2) & LIMB_MASK;
        y_small[i * n] = (uint64_t)(i + 3) & LIMB_MASK;
    }

    std::vector<uint64_t> x_large, y_large;
    make_large(x_large, 2);
    make_large(y_large, 3);

    std::vector<uint64_t> x_rand, y_rand;
    make_rand(x_rand, 0xDEADBEEF12345678ULL);
    make_rand(y_rand, 0xCAFEBABE87654321ULL);

    struct TestPair
    {
        const std::vector<uint64_t> *xa;
        const std::vector<uint64_t> *ya;
        const char *name;
    };
    std::vector<TestPair> pairs = {
        {&x_small, &y_small, "small"},
        {&x_large, &y_large, "large"},
        {&x_rand, &y_rand, "rand "},
    };

    LimbT *d_x, *d_y, *d_out;
    CU(cudaMalloc(&d_x, total_bytes));
    CU(cudaMalloc(&d_y, total_bytes));
    CU(cudaMalloc(&d_out, total_bytes));

    auto print_limbs = [&](const char *label, const uint64_t *v, int cnt)
    {
        printf("      %s: [", label);
        for (int k = 0; k < cnt && k < n; k++)
            printf("%llu ", (unsigned long long)v[k]);
        printf("...]\n");
    };

    static constexpr int NPRINT = 6;

    for (auto &tp : pairs)
    {
        const auto &x_cur = *tp.xa;
        const auto &y_cur = *tp.ya;
        printf("\n  -- values: %s --\n", tp.name);

        std::vector<uint64_t> x_mont, y_mont;
        mont.to_residue_batch(x_cur, x_mont);
        mont.to_residue_batch(y_cur, y_mont);
        CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
        CU(limb_upload(d_y, y_mont.data(), (size_t)nb * n));

        int pass = 0, fail = 0;

        printf("  [roundtrip] ");
        {
            std::vector<uint64_t> rt_result;
            mont.from_residue_batch(d_x, rt_result);
            int rp = 0, rf = 0;
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(rt_result.data() + i * n, x_cur.data() + i * n, n))
                {
                    rp++;
                }
                else
                {
                    rf++;
                    if (rf <= 2)
                    {
                        printf("\n    FAIL cand %d:", i);
                        print_limbs("exp", x_cur.data() + i * n, NPRINT);
                        print_limbs("got", rt_result.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", rp, rp + rf);
            pass += rp;
            fail += rf;
        }

        printf("  [ntt_raw_sq]");
        {
            mont.ntt.ntt_A(d_x, n, 0);
            mont.ntt.psq_and_intt(0);
            CU(cudaDeviceSynchronize());

            int n_sum = mont.n_sum;
            LimbT *d_T_test;
            CU(cudaMalloc(&d_T_test, (size_t)nb * n_sum * sizeof(LimbT)));
            mont.ntt.carry_to_limbs(d_T_test, n_sum, 0);
            CU(cudaDeviceSynchronize());

            std::vector<uint64_t> raw_mul((size_t)nb * n_sum);
            CU(limb_download(raw_mul.data(), d_T_test, raw_mul.size()));

            int rp2 = 0, rf2 = 0;
            for (int i = 0; i < nb && i < 2; i++)
            {
                mpz_t xm, res;
                mpz_init(xm);
                mpz_init(res);
                lims_to_gmp(xm, x_mont.data() + i * n, n);
                mpz_mul(res, xm, xm);
                std::vector<uint64_t> ref2(n_sum, 0);
                gmp_to_lims(ref2.data(), n_sum, res);
                mpz_clear(xm);
                mpz_clear(res);
                if (limbs_eq(raw_mul.data() + i * n_sum, ref2.data(), n_sum))
                {
                    rp2++;
                }
                else
                {
                    rf2++;
                    int first_diff = -1;
                    for (int j = 0; j < n_sum; j++)
                        if (raw_mul[i * n_sum + j] != ref2[j])
                        {
                            first_diff = j;
                            break;
                        }
                    printf("\n    FAIL cand %d raw ntt_sq (first diff at limb %d / %d):", i, first_diff, n_sum);
                    if (first_diff >= 0)
                    {
                        int start = std::max(0, first_diff - 1);
                        printf("\n      expected[%d..]: ", start);
                        for (int j = start; j < std::min(n_sum, start + 8); j++)
                            printf("%llu ", (unsigned long long)ref2[j]);
                        printf("\n      obtained[%d..]: ", start);
                        for (int j = start; j < std::min(n_sum, start + 8); j++)
                            printf("%llu ", (unsigned long long)raw_mul[i * n_sum + j]);
                        printf("\n");
                    }
                }
            }
            printf("%d/2 OK\n", rp2);
            CU(cudaFree(d_T_test));
            CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
        }

        pass = 0;
        fail = 0;
        printf("  [mont_sq]  ");
        mont.modsq_batch(d_x, d_out);
        CU(cudaDeviceSynchronize());
        {
            std::vector<uint64_t> sq_result;
            mont.from_residue_batch(d_out, sq_result);
            for (int i = 0; i < nb; i++)
            {
                std::vector<uint64_t> ref(n, 0);
                gmp_sq_mod(ref.data(), x_cur.data() + i * n, N_h.data() + i * n, n);
                if (limbs_eq(sq_result.data() + i * n, ref.data(), n))
                {
                    pass++;
                }
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d (x=%llu):", i, (unsigned long long)x_cur[i * n]);
                        print_limbs("expected", ref.data(), NPRINT);
                        print_limbs("obtained", sq_result.data() + i * n, NPRINT);
                        std::vector<uint64_t> raw_out((size_t)nb * n);
                        CU(limb_download(raw_out.data(), d_out, raw_out.size()));
                        print_limbs("raw GPU ", raw_out.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [mont_mul] ");
        CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
        mont.modmul_batch(d_x, d_y, d_out);
        CU(cudaDeviceSynchronize());
        {
            std::vector<uint64_t> mul_result;
            mont.from_residue_batch(d_out, mul_result);
            for (int i = 0; i < nb; i++)
            {
                std::vector<uint64_t> ref(n, 0);
                gmp_mul_mod(ref.data(), x_cur.data() + i * n, y_cur.data() + i * n, N_h.data() + i * n, n);
                if (limbs_eq(mul_result.data() + i * n, ref.data(), n))
                {
                    pass++;
                }
                else
                {
                    fail++;
                    if (fail <= 2)
                        printf("\n    FAIL cand %d: x=%llu y=%llu, expected x*y%%N != obtained",
                               i, (unsigned long long)x_cur[i * n], (unsigned long long)y_cur[i * n]);
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [sq x16]   ");
        {
            CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
            LimbT *cur = d_x;
            LimbT *tmp = d_out;
            for (int k = 0; k < 16; k++)
            {
                mont.modsq_batch(cur, tmp);
                std::swap(cur, tmp);
            }
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> res_gpu;
            mont.from_residue_batch(cur, res_gpu);
            for (int i = 0; i < nb; i++)
            {
                mpz_t xm, Nm, res;
                mpz_init(xm);
                mpz_init(Nm);
                mpz_init(res);
                lims_to_gmp(xm, x_cur.data() + i * n, n);
                lims_to_gmp(Nm, N_h.data() + i * n, n);
                mpz_set(res, xm);
                for (int k = 0; k < 16; k++)
                {
                    mpz_mul(res, res, res);
                    mpz_mod(res, res, Nm);
                }
                std::vector<uint64_t> ref(n, 0);
                gmp_to_lims(ref.data(), n, res);
                mpz_clear(xm);
                mpz_clear(Nm);
                mpz_clear(res);
                if (limbs_eq(res_gpu.data() + i * n, ref.data(), n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d", i);
                        print_limbs("exp", ref.data(), NPRINT);
                        print_limbs("gpu", res_gpu.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [sq==mul]  ");
        {
            CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
            CU(limb_upload(d_y, x_mont.data(), (size_t)nb * n));
            LimbT *d_sq_out;
            CU(cudaMalloc(&d_sq_out, total_bytes));
            mont.modsq_batch(d_x, d_out);
            mont.modmul_batch(d_x, d_y, d_sq_out);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> sq_r, mul_r;
            mont.from_residue_batch(d_out, sq_r);
            mont.from_residue_batch(d_sq_out, mul_r);
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(sq_r.data() + i * n, mul_r.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: sq!=mul", i);
                        print_limbs("sq ", sq_r.data() + i * n, NPRINT);
                        print_limbs("mul", mul_r.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_sq_out);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [(N-1)^2]  ");
        {
            std::vector<uint64_t> nm1_all((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
            {
                const uint64_t *Ni = N_h.data() + i * n;
                uint64_t *out2 = nm1_all.data() + i * n;
                std::copy(Ni, Ni + n, out2);
                for (int j = 0; j < n; j++)
                {
                    if (out2[j] > 0)
                    {
                        out2[j]--;
                        break;
                    }
                    out2[j] = LIMB_MASK;
                }
            }
            std::vector<uint64_t> nm1_mont;
            mont.to_residue_batch(nm1_all, nm1_mont);
            CU(limb_upload(d_x, nm1_mont.data(), (size_t)nb * n));
            mont.modsq_batch(d_x, d_out);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> nm1sq;
            mont.from_residue_batch(d_out, nm1sq);
            for (int i = 0; i < nb; i++)
            {
                std::vector<uint64_t> one(n, 0);
                one[0] = 1;
                if (limbs_eq(nm1sq.data() + i * n, one.data(), n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: (N-1)^2%%N != 1", i);
                        print_limbs("got", nm1sq.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [commut]   ");
        {
            CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
            CU(limb_upload(d_y, y_mont.data(), (size_t)nb * n));
            LimbT *d_xy, *d_yx;
            CU(cudaMalloc(&d_xy, total_bytes));
            CU(cudaMalloc(&d_yx, total_bytes));
            mont.modmul_batch(d_x, d_y, d_xy);
            mont.modmul_batch(d_y, d_x, d_yx);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> xy_h((size_t)nb * n), yx_h((size_t)nb * n);
            CU(limb_download(xy_h.data(), d_xy, xy_h.size()));
            CU(limb_download(yx_h.data(), d_yx, yx_h.size()));
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(xy_h.data() + i * n, yx_h.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                        printf("\n    FAIL cand %d: xy!=yx", i);
                }
            }
            cudaFree(d_xy);
            cudaFree(d_yx);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [identity] ");
        {
            std::vector<uint64_t> one_plain((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
                one_plain[i * n] = 1;
            std::vector<uint64_t> one_mont_v;
            mont.to_residue_batch(one_plain, one_mont_v);
            LimbT *d_one2;
            CU(cudaMalloc(&d_one2, total_bytes));
            CU(limb_upload(d_x, x_mont.data(), (size_t)nb * n));
            CU(limb_upload(d_one2, one_mont_v.data(), (size_t)nb * n));
            mont.modmul_batch(d_x, d_one2, d_out);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> id_res;
            mont.from_residue_batch(d_out, id_res);
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(id_res.data() + i * n, x_cur.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: x*1 != x", i);
                        print_limbs("exp", x_cur.data() + i * n, NPRINT);
                        print_limbs("got", id_res.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_one2);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [modpow32] ");
        {
            const uint32_t EXP32 = 0xDEADBEEFU;
            std::vector<uint64_t> one_plain((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
                one_plain[i * n] = 1;
            std::vector<uint64_t> one_m;
            mont.to_residue_batch(one_plain, one_m);
            LimbT *d_acc, *d_base32, *d_tmp32;
            CU(cudaMalloc(&d_acc, total_bytes));
            CU(cudaMalloc(&d_base32, total_bytes));
            CU(cudaMalloc(&d_tmp32, total_bytes));
            CU(limb_upload(d_acc, one_m.data(), (size_t)nb * n));
            CU(limb_upload(d_base32, x_mont.data(), (size_t)nb * n));
            for (int bit = 31; bit >= 0; bit--)
            {
                mont.modsq_batch(d_acc, d_tmp32);
                std::swap(d_acc, d_tmp32);
                if ((EXP32 >> bit) & 1)
                {
                    mont.modmul_batch(d_acc, d_base32, d_tmp32);
                    std::swap(d_acc, d_tmp32);
                }
            }
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> modpow_gpu;
            mont.from_residue_batch(d_acc, modpow_gpu);
            for (int i = 0; i < nb; i++)
            {
                mpz_t xm, Nm, res;
                mpz_init(xm);
                mpz_init(Nm);
                mpz_init(res);
                lims_to_gmp(xm, x_cur.data() + i * n, n);
                lims_to_gmp(Nm, N_h.data() + i * n, n);
                mpz_powm_ui(res, xm, (unsigned long)EXP32, Nm);
                std::vector<uint64_t> ref(n, 0);
                gmp_to_lims(ref.data(), n, res);
                mpz_clear(xm);
                mpz_clear(Nm);
                mpz_clear(res);
                if (limbs_eq(modpow_gpu.data() + i * n, ref.data(), n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: x^0xDEADBEEF%%N wrong", i);
                        print_limbs("exp", ref.data(), NPRINT);
                        print_limbs("got", modpow_gpu.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_acc);
            cudaFree(d_base32);
            cudaFree(d_tmp32);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        pass = 0;
        fail = 0;
        printf("  [window_ok] ");
        {
            const uint32_t EXP32W = 0xCAFEBABEU;
            std::vector<uint64_t> one_plain2((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
                one_plain2[i * n] = 1;
            std::vector<uint64_t> one_m2;
            mont.to_residue_batch(one_plain2, one_m2);
            LimbT *d_ref_acc, *d_win_acc, *d_base_w, *d_tmp_w;
            CU(cudaMalloc(&d_ref_acc, total_bytes));
            CU(cudaMalloc(&d_win_acc, total_bytes));
            CU(cudaMalloc(&d_base_w, total_bytes));
            CU(cudaMalloc(&d_tmp_w, total_bytes));
            CU(limb_upload(d_ref_acc, one_m2.data(), (size_t)nb * n));
            CU(limb_upload(d_base_w, x_mont.data(), (size_t)nb * n));
            for (int bit = 31; bit >= 0; bit--)
            {
                mont.modsq_batch(d_ref_acc, d_tmp_w);
                std::swap(d_ref_acc, d_tmp_w);
                if ((EXP32W >> bit) & 1)
                {
                    mont.modmul_batch(d_ref_acc, d_base_w, d_tmp_w);
                    std::swap(d_ref_acc, d_tmp_w);
                }
            }
            {
                constexpr int K = 4, SZ = 1 << K;
                std::vector<LimbT *> tbl(SZ);
                for (int w = 0; w < SZ; w++)
                    CU(cudaMalloc(&tbl[w], total_bytes));
                CU(limb_upload(tbl[0], one_m2.data(), (size_t)nb * n));
                CU(limb_upload(tbl[1], x_mont.data(), (size_t)nb * n));
                for (int w = 2; w < SZ; w++)
                    mont.modmul_batch(tbl[w - 1], d_base_w, tbl[w]);
                LimbT *d_table_w;
                CU(cudaMalloc(&d_table_w, (size_t)SZ * total_bytes));
                for (int w = 0; w < SZ; w++)
                    CU(cudaMemcpy(d_table_w + (size_t)w * nb * n, tbl[w], total_bytes, cudaMemcpyDeviceToDevice));
                std::vector<uint64_t> exp32w_all((size_t)nb * n, 0);
                for (int i = 0; i < nb; i++)
                {
                    uint32_t e = EXP32W;
                    for (int j = 0; j < (int)((32 + LIMB_BITS - 1) / LIMB_BITS) && j < n; j++)
                    {
                        exp32w_all[i * n + j] = e & LIMB_MASK;
                        e >>= LIMB_BITS;
                    }
                }
                const size_t exp_bytes = (size_t)nb * n * sizeof(Data64);
                Data64 *d_exp_w;
                CU(cudaMalloc(&d_exp_w, exp_bytes));
                CU(cudaMemcpy(d_exp_w, exp32w_all.data(), exp_bytes, cudaMemcpyHostToDevice));
                LimbT *d_cur_w;
                CU(cudaMalloc(&d_cur_w, total_bytes));
                CU(limb_upload(d_win_acc, one_m2.data(), (size_t)nb * n));
                const int thr = 256;
                dim3 gsel((unsigned)(n + thr - 1) / thr, (unsigned)nb);
                int n_win32 = (32 + K - 1) / K;
                int start32 = n_win32 * K - 1;
                for (int win = 0; win < n_win32; win++)
                {
                    int msb_pos = start32 - win * K;
                    for (int sq = 0; sq < K; sq++)
                    {
                        mont.modsq_batch(d_win_acc, d_tmp_w);
                        std::swap(d_win_acc, d_tmp_w);
                    }
                    select_window_kernel<<<gsel, thr>>>(d_cur_w, d_table_w, d_exp_w, msb_pos, K, n, nb);
                    mont.modmul_batch(d_win_acc, d_cur_w, d_tmp_w);
                    std::swap(d_win_acc, d_tmp_w);
                }
                CU(cudaDeviceSynchronize());
                for (int w = 0; w < SZ; w++)
                    cudaFree(tbl[w]);
                cudaFree(d_table_w);
                cudaFree(d_exp_w);
                cudaFree(d_cur_w);
            }
            std::vector<uint64_t> ref_r, win_r;
            mont.from_residue_batch(d_ref_acc, ref_r);
            mont.from_residue_batch(d_win_acc, win_r);
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(ref_r.data() + i * n, win_r.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: window k=4 != sq-mul", i);
                        print_limbs("sq-mul", ref_r.data() + i * n, NPRINT);
                        print_limbs("window", win_r.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_ref_acc);
            cudaFree(d_win_acc);
            cudaFree(d_base_w);
            cudaFree(d_tmp_w);
            printf("%d/%d OK\n", pass, pass + fail);
        }

    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    printf("\n=== End of tests ===\n\n");
}

// ── Standalone entry point (no input file needed) ────────────────────
//
// Builds a small BatchModCtx from a fixed set of known primes so that
// run_correctness_tests can be called without any candidate file.

void run_correctness_tests()
{
    constexpr int NB = 512;
    constexpr int TARGET_DIGITS = 5000;
    constexpr int TARGET_BITS = (int)(TARGET_DIGITS / 0.30103) + 1;

    int n_limbs = limbs_for_digits(TARGET_DIGITS + 4);

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    unsigned long seed = 0;
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom)
    {
        if (fread(&seed, sizeof(seed), 1, urandom) != 1)
            seed = (unsigned long)time(nullptr);
        fclose(urandom);
    }
    gmp_randseed_ui(rng, seed);

    std::vector<NumberCandidate> cands(NB);
    for (int i = 0; i < NB; i++)
    {
        mpz_t N;
        mpz_init(N);
        mpz_urandomb(N, rng, TARGET_BITS);
        mpz_setbit(N, TARGET_BITS - 1);
        mpz_setbit(N, 0);
        cands[i].build_from_mpz(N, n_limbs);
        mpz_clear(N);
    }
    gmp_randclear(rng);

    std::vector<NumberCandidate *> ptrs(NB);
    for (int i = 0; i < NB; i++)
        ptrs[i] = &cands[i];

    std::vector<uint64_t> N_all, Nm1_all, d_all;
    pack_batch(ptrs, n_limbs, N_all, Nm1_all, d_all);
    BatchModCtx ctx(N_all, n_limbs, NB);
    run_ops_against_gmp(ctx, N_all);
}
