/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_ops.cu
 * ROLE   driver and report of --bench-ops
 *
 * HOW    Sweeps the operand sizes, calls both sides for every row and
 *        prints the comparison table in ops/day. The measurements
 *        themselves live in bench_ops_gmp.cu and bench_ops_gpu.cu.
 *
 * CHANGELOG
 *   2026-08-11  Reduced to the driver; the measurements moved to
 *               bench_ops_{gmp,gpu}.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "bench/bench_ops.cuh"
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

void run_bench_ops(bool long_run)
{
    const int *BIT_SIZES = long_run ? BIT_SIZES_LONG : BIT_SIZES_SHORT;
    const int N_SIZES = long_run ? (int)(sizeof(BIT_SIZES_LONG) / sizeof(BIT_SIZES_LONG[0]))
                                 : (int)(sizeof(BIT_SIZES_SHORT) / sizeof(BIT_SIZES_SHORT[0]));


    const char *row_names[] = {
        "GPU mul         (ops/d)",
        "GPU sq          (ops/d)",
        "GPU mont_mul    (ops/d)",
        "GPU sq+mod      (ops/d)",
        "GPU miller-rabin(ops/d)",
        "GMP mul         (ops/d)",
        "GMP sq          (ops/d)",
        "GMP mul+mod     (ops/d)",
        "GMP sq+mod      (ops/d)",
        "GMP miller-rabin(ops/d)",
    };
    const int N_ROWS = 10;

    const int speedup_pairs[][2] = {{0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}};
    const char *speedup_names[] = {
        "Speedup GPU/GMP mul     ",
        "Speedup GPU/GMP sq      ",
        "Speedup GPU/GMP mul+mod ",
        "Speedup GPU/GMP sq+mod  ",
        "Speedup GPU/GMP MR      ",
    };

    BenchResult results[N_ROWS][N_SIZES] = {};

    printf("=== Benchmark MillerRabin-GPU vs GMP  (batch=%d, %.0fs/test) ===\n"
           "    GPU mul/sq: NTT+pmul+INTT without REDC  |  mul+mod/sq: with REDC\n"
           "    GMP mul/sq: multiplication only          |  mul+mod/sq+mod: with reduction\n"
           "    MR: 1 witness {2}, N equiv 3 mod 4 (s=1)\n\n",
           N_BATCH, BENCH_SECS);

    for (int c = 0; c < N_SIZES; c++)
    {
        int bits = BIT_SIZES[c];
        bool last = (c == N_SIZES - 1);
        printf("── %d bits ──\n", bits);

        printf("  GPU mul          ... ");
        fflush(stdout);
        results[0][c] = bench_gpu_mul_only(bits, last);
        printf("%s\n", fmt_time(results[0][c].elapsed_sec).c_str());

        printf("  GPU sq           ... ");
        fflush(stdout);
        results[1][c] = bench_gpu_sq_only(bits, last);
        printf("%s\n", fmt_time(results[1][c].elapsed_sec).c_str());

        printf("  GPU mul+mod     ... ");
        fflush(stdout);
        results[2][c] = bench_gpu_mul(bits, last);
        printf("%s\n", fmt_time(results[2][c].elapsed_sec).c_str());

        printf("  GPU sq+mod      ... ");
        fflush(stdout);
        results[3][c] = bench_gpu_sq(bits, last);
        printf("%s\n", fmt_time(results[3][c].elapsed_sec).c_str());

        printf("  GPU miller-rabin ... ");
        fflush(stdout);
        results[4][c] = bench_gpu_mr(bits, last);
        printf("%s\n", fmt_time(results[4][c].elapsed_sec).c_str());

        printf("  GMP mul          ... ");
        fflush(stdout);
        results[5][c] = bench_gmp_mul_only(bits, last);
        printf("%s\n", fmt_time(results[5][c].elapsed_sec).c_str());

        printf("  GMP sq           ... ");
        fflush(stdout);
        results[6][c] = bench_gmp_sq_only(bits, last);
        printf("%s\n", fmt_time(results[6][c].elapsed_sec).c_str());

        printf("  GMP mul+mod      ... ");
        fflush(stdout);
        results[7][c] = bench_gmp_mul(bits, last);
        printf("%s\n", fmt_time(results[7][c].elapsed_sec).c_str());

        printf("  GMP sq+mod       ... ");
        fflush(stdout);
        results[8][c] = bench_gmp_sq(bits, last);
        printf("%s\n", fmt_time(results[8][c].elapsed_sec).c_str());

        printf("  GMP miller-rabin ... ");
        fflush(stdout);
        results[9][c] = bench_gmp_mr(bits, last);
        printf("%s\n\n", fmt_time(results[9][c].elapsed_sec).c_str());
    }

    const int COL_W = 16;
    const int ROW_W = 28;

    printf("\n");
    printf("%-*s", ROW_W, "Operation (ops/day)");
    for (int c = 0; c < N_SIZES; c++)
        printf("  %*d-bit", COL_W - 5, BIT_SIZES[c]);
    printf("\n%s\n", std::string(ROW_W + N_SIZES * COL_W, '-').c_str());

    for (int r = 0; r < N_ROWS; r++)
    {
        printf("%-*s", ROW_W, row_names[r]);
        for (int c = 0; c < N_SIZES; c++)
        {
            const auto &res = results[r][c];
            printf("  %*s", COL_W - 2,
                   res.skipped ? "N/A" : fmt_ops_per_day(res.ops_per_sec).c_str());
        }
        printf("\n");
        if (r == 4)
            printf("\n");
    }

    printf("%s\n", std::string(ROW_W + N_SIZES * COL_W, '-').c_str());
    for (int p = 0; p < 5; p++)
    {
        int gi = speedup_pairs[p][0], mi = speedup_pairs[p][1];
        printf("%-*s", ROW_W, speedup_names[p]);
        for (int c = 0; c < N_SIZES; c++)
        {
            if (results[gi][c].skipped || results[mi][c].ops_per_sec <= 0)
            {
                printf("  %*s", COL_W - 2, "N/A");
            }
            else
            {
                char buf[32];
                snprintf(buf, sizeof(buf), "%6.1fx",
                         results[gi][c].ops_per_sec / results[mi][c].ops_per_sec);
                printf("  %*s", COL_W - 2, buf);
            }
        }
        printf("\n");
    }

    printf("\n");

}
