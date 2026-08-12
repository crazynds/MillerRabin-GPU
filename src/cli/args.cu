/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/cli/args.cu
 * ROLE   command-line parsing
 *
 * HOW    Straight scan over argv. Unrecognised words are taken as the input
 *        file, so the file may appear anywhere in the line.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_mr_gpu.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "cli/options.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <algorithm>

int parse_args(int argc, char *argv[], Options &o)
{
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--test")
            o.run_tests = true;
        else if (a == "--report")
            o.show_report = true;
        else if (a == "--progress")
            o.show_progress = true;
        else if (a == "--bench-ops")
            o.run_bench = true;
        else if (a == "--bench-ops-long")
            o.run_bench_long = true;
        else if (a == "--config")
            o.show_config = true;
        else if (a == "--help" || a == "-h")
            o.show_help = true;
        else if (a == "--cpu")
            o.cpu_mode = true;
        else if (a == "--bench-compare")
            o.run_bench_compare = true;
        else if (a == "--digits" && i + 1 < argc)
            o.cmp.digits = std::max(1, atoi(argv[++i]));
        else if (a == "--timeout" && i + 1 < argc)
            o.cmp.throughput_timeout_s = std::max(1, atoi(argv[++i]));
        else if (a == "--single-iters" && i + 1 < argc)
            o.cmp.single_iters = std::max(1, atoi(argv[++i]));
        else if (a == "--skip-phase-1")
            o.cmp.skip_phase1 = true;
        else if (a == "--skip-phase-2")
            o.cmp.skip_phase2 = true;
        else if (a.rfind("--do=", 0) == 0)
        {
            std::string who = a.substr(5);
            if (who == "gpu")
            {
                o.cmp.gpu_only = true;
                o.cmp.cpu_only = false;
            }
            else
            {
                size_t suffix_pos = who.rfind("cpu");
                if (suffix_pos == std::string::npos || suffix_pos == 0)
                {
                    fprintf(stderr, "Invalid --do value '%s' (expected \"gpu\" or \"<N>cpu\")\n", who.c_str());
                    return 1;
                }
                int n_threads = std::max(1, atoi(who.substr(0, suffix_pos).c_str()));
                o.cmp.cpu_only = true;
                o.cmp.gpu_only = false;
                o.cmp.cpu_threads_fixed = n_threads;
            }
        }
        else if (a == "--cpu-parallel")
        {
            o.cpu_mode = true;
            if (o.cpu_threads == 0)
                o.cpu_threads = (int)std::thread::hardware_concurrency();
        }
        else if ((a == "--threads" || a == "-j") && i + 1 < argc)
        {
            o.cpu_threads = std::max(1, atoi(argv[++i]));
            o.cpu_mode = true;
        }
        else if (!o.input_file)
            o.input_file = argv[i];
    }
    return 0;
}
