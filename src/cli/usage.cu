/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/cli/usage.cu
 * ROLE   the --help text
 *
 * HOW    A single printf of the synopsis and every flag. Kept apart from
 *        parsing so adding a flag touches two obvious places instead of one
 *        370-line function.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_mr_gpu.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "cli/options.h"
#include <cstdio>

void print_usage(const char *prog, FILE *out)
{
    fprintf(out,
        "Usage: %s [options] <input.txt>\n"
        "\n"
        "Input format — one equation per line, with an optional group prefix:\n"
        "\n"
        "  [group_id:] equation\n"
        "\n"
        "  group_id  any string without ':'. Optional. Lines sharing a group_id\n"
        "            are tested together: if one equation is composite, the\n"
        "            rest of the group is skipped.\n"
        "  equation  integer arithmetic expression: + - * / %% ^ and parentheses,\n"
        "            e.g. \"10^18001 - 25*10^1334 - 1\"\n"
        "\n"
        "  Lines with no ':' are treated as a singleton group. Blank lines and\n"
        "  lines beginning with '#' are ignored.\n"
        "\n"
        "Options:\n"
        "  --test              Run GMP-checked correctness tests before the benchmark.\n"
        "  --report            Print a per-candidate detail report.\n"
        "  --progress          Show a live GPU progress bar.\n"
        "  --config            Print the active build configuration and exit.\n"
        "  --bench-ops         Benchmark the individual GPU primitives.\n"
        "  --bench-ops-long    Longer/more thorough primitive benchmark.\n"
        "  --cpu               Use GMP mpz_probab_prime_p (CPU) instead of GPU; same\n"
        "                      group semantics, one candidate at a time, no batching.\n"
        "  --cpu-parallel      Like --cpu, but spread across all hardware threads.\n"
        "  --threads N, -j N   Like --cpu, but spread across N threads.\n"
        "  --help, -h          Show this help and exit.\n"
        "\n"
        "  --bench-compare     Self-benchmark: GPU vs CPU on a freshly generated,\n"
        "                      small-factor-free candidate. Ignores <input.txt>. CPU\n"
        "                      throughput is swept across every thread count from 1\n"
        "                      to the machine's hardware concurrency automatically.\n"
        "\n"
        "  Options for --bench-compare:\n"
        "    --digits N        decimal digits of the test candidate (default 100000)\n"
        "    --timeout S       seconds per run for the throughput phase (default 30)\n"
        "    --single-iters K  latency-phase repetitions per side (default 10)\n"
        "    --skip-phase-1    skip the throughput phase\n"
        "    --skip-phase-2    skip the single-candidate latency phase\n"
        "    --do=WHO          restrict --bench-compare to one side, printing its\n"
        "                      results as soon as each phase finishes (useful for\n"
        "                      firing one sbatch job per side):\n"
        "                        --do=gpu     GPU only\n"
        "                        --do=Ncpu    CPU only, with exactly N threads,\n"
        "                                     e.g. --do=1cpu, --do=8cpu\n"
        "                      Without --do, GPU runs once and CPU is swept across\n"
        "                      every thread count from 1 to hardware_concurrency().\n",
        prog);
}
