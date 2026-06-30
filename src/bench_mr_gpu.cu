// bench_mr_gpu.cu — Miller-Rabin GPU benchmark driver.
//
// Usage: ./bench_mr_gpu [options] <input.txt>
//
// Input format — one equation per line, with an optional group prefix:
//
//   [group_id:] equation
//
//   group_id  — any string without ':' (integers, letters, etc.). Optional.
//               Lines that share the same group_id are tested together: each
//               equation is tested in order; if any is composite the rest are
//               skipped (the group fails). Only when every equation in the group
//               passes does the group count as a winner.
//   equation  — integer arithmetic expression (see equation.h for the grammar).
//               Supports + - * / % ^ and parentheses; e.g. "10^18001 - 25*10^1334 - 1"
//
//   Lines with no ':' are treated as a group of their own (singleton group).
//   Blank lines and lines beginning with '#' are ignored.
//
// Options:
//   --test            Run GMP-checked correctness tests before the benchmark.
//   --report          Print a per-candidate detail report.
//   --progress        Show a live GPU progress bar.
//   --config          Print the active build configuration and exit.
//   --bench-ops       Benchmark the individual GPU primitives.
//   --bench-ops-long  Longer/more thorough primitive benchmark.
//   --cpu             Use GMP mpz_probab_prime_p (CPU) instead of GPU; same group
//                     semantics, one candidate at a time, no batching.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <thread>
#include <cuda_runtime.h>

#include "candidate.cuh"
#include "miller_rabin_runner.cuh"
#include "correctness_tests.cuh"
#include "helpers/bench_ops.cuh"
#include "input_parser.h"
#include "cpu_runner.h"

using hrc = std::chrono::high_resolution_clock;

static constexpr int BATCH_SIZE = MR_BATCH_SIZE;

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    bool run_tests = false;
    bool show_report = false;
    bool show_progress = false;
    bool run_bench = false;
    bool run_bench_long = false;
    bool show_config = false;
    bool cpu_mode = false;
    int  cpu_threads = 0;
    const char *input_file = nullptr;

    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--test")
            run_tests = true;
        else if (a == "--report")
            show_report = true;
        else if (a == "--progress")
            show_progress = true;
        else if (a == "--bench-ops")
            run_bench = true;
        else if (a == "--bench-ops-long")
            run_bench_long = true;
        else if (a == "--config")
            show_config = true;
        else if (a == "--cpu")
            cpu_mode = true;
        else if (a == "--cpu-parallel")
        {
            cpu_mode = true;
            if (cpu_threads == 0)
                cpu_threads = (int)std::thread::hardware_concurrency();
        }
        else if ((a == "--threads" || a == "-j") && i + 1 < argc)
        {
            cpu_threads = std::max(1, atoi(argv[++i]));
            cpu_mode = true;
        }
        else if (!input_file)
            input_file = argv[i];
    }

    if (show_config)
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

    if (run_bench || run_bench_long)
    {
        run_bench_ops(run_bench_long);
        return 0;
    }

    if (run_tests && !input_file)
    {
        run_correctness_tests();
        run_known_prime_tests();
        run_general_s_prime_tests();
        run_s1_nextprime_tests();
        return 0;
    }

    if (!input_file)
    {
        fprintf(stderr,
                "Usage: %s [--test] [--report] [--progress] [--config]"
                " [--bench-ops] [--bench-ops-long] [--cpu] [--cpu-parallel]"
                " [--threads N | -j N] <input.txt>\n",
                argv[0]);
        return 1;
    }

    // ── Load candidates ───────────────────────────────────────────────────────
    std::vector<GroupInfo> groups;
    try
    {
        groups = parse_input(input_file);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error reading input: %s\n", e.what());
        return 1;
    }
    if (groups.empty())
    {
        fprintf(stderr, "No candidates found in %s\n", input_file);
        return 1;
    }

    // ── CPU mode ──────────────────────────────────────────────────────────────
    if (cpu_mode)
    {
        int n_threads = (cpu_threads > 0) ? cpu_threads : 1;
        run_cpu_mode(groups, show_report, show_progress, n_threads);
        return 0;
    }

    int n_groups = (int)groups.size();
    int max_rounds = 0;
    for (auto &g : groups)
        if ((int)g.equations.size() > max_rounds)
            max_rounds = (int)g.equations.size();

    {
        int total_eqs = 0;
        for (auto &g : groups)
            total_eqs += (int)g.equations.size();
        printf("Groups: %d  total equations: %d  max rounds: %d\n",
               n_groups, total_eqs, max_rounds);
        printf("Batch size: %d (sub-batch)  witnesses: %d\n\n",
               BATCH_SIZE, (int)DEFAULT_WITNESSES.size());
    }

    // ── Correctness tests ─────────────────────────────────────────────────────
    if (run_tests)
    {
        run_correctness_tests();
        run_known_prime_tests();
        run_general_s_prime_tests();
        run_s1_nextprime_tests();
    }

    // ── Round-based GPU testing ───────────────────────────────────────────────
    //
    // alive[i] = true  → group i is still in the running.
    // In each round, ALL still-alive groups dispatch their round-R equation at once.
    // The MR runner tests with witness 1 across ALL candidates (in sub-batches of
    // BATCH_SIZE), then compacts globally, then proceeds to witness 2, etc.
    // This way witness 2 only processes the global survivors of witness 1.

    auto t_global = hrc::now();
    std::vector<bool> alive(n_groups, true);

    for (int round = 0; round < max_rounds; round++)
    {
        // Collect groups that still have a candidate for this round
        std::vector<int> active;
        for (int gi = 0; gi < n_groups; gi++)
            if (alive[gi] && round < (int)groups[gi].equations.size())
                active.push_back(gi);

        if (active.empty())
            break;

        int n_active = (int)active.size();
        printf("\n=== Round %d (%d groups active) ===\n", round + 1, n_active);
        fflush(stdout);
        auto t_round = hrc::now();

        // Build LazyCandidate stubs for this round; evaluate mpz (fast) to find max n_limbs.
        // Limb arrays are NOT built yet — only computed on demand per sub-batch below.
        std::vector<LazyCandidate> lcs(n_active);
        int batch_n_limbs = 0;
        for (int k = 0; k < n_active; k++)
        {
            int gi = active[k];
            lcs[k].equation  = groups[gi].equations[round];
            lcs[k].group_idx = gi;
            lcs[k].round_idx = round;
            int nl = lcs[k].natural_n_limbs();   // mpz eval only, no limb build
            if (nl > batch_n_limbs) batch_n_limbs = nl;
        }

        // Global witness sweep with lazy build.
        // alive_idx[i] = index into lcs[] (and active[]).
        // For each witness: test all alive candidates in sub-batches of BATCH_SIZE,
        // building limbs for each sub-batch just before the GPU call.
        std::vector<int> alive_idx(n_active);
        for (int i = 0; i < n_active; i++) alive_idx[i] = i;

        for (uint32_t w : DEFAULT_WITNESSES)
        {
            if (alive_idx.empty()) break;

            int n_alive = (int)alive_idx.size();
            if (show_report)
            {
                printf("  [r%d] Witness %-3u  alive: %d\n", round + 1, w, n_alive);
                fflush(stdout);
            }

            std::vector<bool> passed_this_witness(n_alive, false);

            for (int start = 0; start < n_alive; start += BATCH_SIZE)
            {
                int sub_bsz = std::min(BATCH_SIZE, n_alive - start);

                // Build limbs lazily — only for this sub-batch
                int s = 1;
                std::vector<const NumberCandidate *> sub_cands;
                sub_cands.reserve(sub_bsz);
                for (int i = 0; i < sub_bsz; i++)
                {
                    const auto &nc = lcs[alive_idx[start + i]].get(batch_n_limbs);
                    if (nc.s > s) s = nc.s;
                    sub_cands.push_back(&nc);
                }

                std::vector<uint64_t> N_sub, Nm1_sub, d_sub;
                pack_batch(sub_cands, batch_n_limbs, N_sub, Nm1_sub, d_sub);

                auto sub_res = gpu_test_witness(N_sub, d_sub, Nm1_sub,
                                                batch_n_limbs, sub_bsz,
                                                s, w, show_progress);
                for (int i = 0; i < sub_bsz; i++)
                    passed_this_witness[start + i] = sub_res[i];
            }

            // Global compact
            std::vector<int> new_alive;
            for (int i = 0; i < n_alive; i++)
            {
                if (passed_this_witness[i])
                    new_alive.push_back(alive_idx[i]);
                else if (show_report)
                    printf("    [r%d] entry %d COMPOSITE (witness %u)\n",
                           round + 1, alive_idx[i], w);
            }
            alive_idx = std::move(new_alive);
        }

        // Map survivors back to groups
        int survivors = 0;
        std::vector<bool> results(n_active, false);
        for (int idx : alive_idx) results[idx] = true;
        for (int k = 0; k < n_active; k++)
        {
            int gi = active[k];
            if (!results[k])
                alive[gi] = false;
            else
                survivors++;
        }

        double t_r = std::chrono::duration_cast<std::chrono::milliseconds>(
                         hrc::now() - t_round).count() / 1000.0;
        printf("Round %d: %.2fs  —  %d survived, %d eliminated\n",
               round + 1, t_r, survivors, n_active - survivors);
    }

    double total = std::chrono::duration_cast<std::chrono::milliseconds>(
                       hrc::now() - t_global).count() / 1000.0;

    // ── Results ───────────────────────────────────────────────────────────────
    printf("\n=== Results ===\n");
    int n_winners = 0;
    for (int gi = 0; gi < n_groups; gi++)
    {
        if (alive[gi])
        {
            n_winners++;
            auto &g = groups[gi];
            if (g.label.empty() || g.label.rfind("__auto_", 0) == 0)
                printf("  PRIME: %s\n", g.equations[0].c_str());
            else
            {
                printf("  PRIME group [%s]:\n", g.label.c_str());
                for (auto &eq : g.equations)
                    printf("    %s\n", eq.c_str());
            }
        }
    }
    if (n_winners == 0)
        printf("  No group passed all rounds.\n");

    printf("Total time: %.2fs\n", total);
    return 0;
}
