/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/cli/config_print.cu
 * ROLE   dump of the compile-time build configuration
 *
 * HOW    Turns the config.h defines back into readable names. Every knob
 *        here is fixed at build time, so this is the only way to tell two
 *        binaries apart at runtime.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_mr_gpu.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "cli/options.h"
#include "config.h"
#include "constants.h"
#include <cstdio>

void print_build_config()
{
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
        const char *carry_alg = "SINGLE_TILE";
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
        const char *carry_alg = "MULTI_TILE";
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
        const char *carry_alg = "PREFIX_SCAN";
#else
        const char *carry_alg = "SEQUENTIAL";
#endif
#if MUL_ALG == MUL_SCHOOLBOOK
        const char *mul_alg = "SCHOOLBOOK";
#elif MUL_ALG == MUL_4STEP_GPUNTT
        const char *mul_alg = "NTT_4STEP";
#else
        const char *mul_alg = "NTT_MERGE";
#endif
#if MOD_REDUCTION_ALG == MOD_RED_MONTGOMERY
        const char *mod_red_alg = "MONTGOMERY";
#elif MOD_REDUCTION_ALG == MOD_RED_BARRETT
        const char *mod_red_alg = "BARRETT";
#else
        const char *mod_red_alg = "BURNIKEL_ZIEGLER";
#endif
        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║  Build configuration                             ║\n");
        printf("╚══════════════════════════════════════════════════╝\n");
        printf("  window_bits       %d\n", MR_WINDOW_BITS);
        printf("  batch_size        %d\n", MR_BATCH_SIZE);
        printf("  mont_mul_alg      %s\n", mul_alg);
        printf("  mod_reduction_alg %s\n", mod_red_alg);
        printf("  carry_norm_alg    %s\n", carry_alg);
        printf("  carry_tile        %d\n", MR_CARRY_TILE);
        printf("  carry_inter_thr   %d\n", MR_CARRY_INTER_THR);
        printf("  thr_load          %d\n", MR_THR_LOAD);
        printf("  thr_pmul          %d\n", MR_THR_PMUL);
        printf("  thr_add_          %d\n", MR_THR_ADD);
        printf("  thr_select_win    %d\n", MR_THR_SELECT_WIN);
        printf("  thr_check         %d\n", MR_THR_CHECK);
        printf("  thr_copy          %d\n", MR_THR_COPY);
        printf("  sub_tile          %d\n", MR_SUB_TILE);
        printf("  progress_interval %d ms\n", MR_PROGRESS_INTERVAL_MS);
#ifdef MR_ADVANCED_MONITOR
        printf("  advanced_monitor  ON\n");
#else
        printf("  advanced_monitor  OFF\n");
#endif
        printf("\n");
}
