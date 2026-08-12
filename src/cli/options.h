/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/cli/options.h
 * ROLE   everything the command line can set
 *
 * HOW    One struct instead of a dozen locals threaded through main().
 *        parse_args fills it and reports whether the program should keep
 *        going.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_mr_gpu.cu, where parsing and running
 *               shared one 370-line main().
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "bench/compare_bench.h"
#include <cstdio>

struct Options
{
    bool run_tests = false;
    bool show_report = false;
    bool show_progress = false;
    bool run_bench = false;
    bool run_bench_long = false;
    bool show_config = false;
    bool show_help = false;
    bool cpu_mode = false;
    bool run_bench_compare = false;
    int cpu_threads = 0;
    const char *input_file = nullptr;
    CompareBenchOptions cmp;
};

/* Fills `out` from argv.
 *
 * PARAMS
 *   argc, argv  as received by main
 *   out         [out] parsed options
 *
 * Returns 0 to continue, or a process exit code when parsing already decided
 * the run is over (bad --do value).
 */
int parse_args(int argc, char *argv[], Options &out);

/* Writes the usage text.
 *
 * PARAMS
 *   prog  argv[0], echoed into the synopsis
 *   out   stdout for --help, stderr for a usage error
 */
void print_usage(const char *prog, FILE *out);

/* Prints the compile-time configuration this binary was built with. */
void print_build_config();
