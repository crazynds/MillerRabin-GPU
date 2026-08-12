/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/main.cu
 * ROLE   entry point: parse, dispatch, done
 *
 * HOW    Every mode is one call. The work lives in cli/ (parsing, usage,
 *        config), tests/ (--test), bench/ (--bench-*) and driver/ (the
 *        actual hunt).
 *
 * CHANGELOG
 *   2026-08-11  Reduced from 486 lines: parsing, usage, config dump and the
 *               round loop moved to cli/ and driver/.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "cli/options.h"
#include "driver/input_parser.h"
#include "driver/round_driver.h"
#include "driver/cpu_runner.h"
#include "bench/bench_ops.cuh"
#include "bench/compare_bench.h"
#include "tests/tests.cuh"
#include <cstdio>
#include <vector>

/* Runs the four correctness suites. */
static void run_all_tests()
{
    run_correctness_tests();
    run_known_prime_tests();
    run_general_s_prime_tests();
    run_s1_nextprime_tests();
}

int main(int argc, char *argv[])
{
    Options o;
    if (int rc = parse_args(argc, argv, o))
        return rc;

    if (o.show_help)
    {
        print_usage(argv[0], stdout);
        return 0;
    }
    if (o.show_config)
        print_build_config();

    if (o.run_bench || o.run_bench_long)
    {
        run_bench_ops(o.run_bench_long);
        return 0;
    }
    if (o.run_bench_compare)
    {
        run_compare_bench(o.cmp);
        return 0;
    }

    if (o.run_tests && !o.input_file)
    {
        run_all_tests();
        return 0;
    }
    if (!o.input_file)
    {
        print_usage(argv[0], stderr);
        return 1;
    }

    std::vector<GroupInfo> groups;
    try
    {
        groups = parse_input(o.input_file);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error reading input: %s\n", e.what());
        return 1;
    }
    if (groups.empty())
    {
        fprintf(stderr, "No candidates found in %s\n", o.input_file);
        return 1;
    }

    if (o.cpu_mode)
    {
        run_cpu_mode(groups, o.show_report, o.show_progress,
                     o.cpu_threads > 0 ? o.cpu_threads : 1);
        return 0;
    }

    if (o.run_tests)
        run_all_tests();

    run_rounds(groups, o.show_report, o.show_progress);
    return 0;
}
