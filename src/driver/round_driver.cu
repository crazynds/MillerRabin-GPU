/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/driver/round_driver.cu
 * ROLE   round-based orchestration of a candidate file
 *
 * HOW    A group is a list of equations that stand or fall together. Each
 *        round dispatches the round-R equation of every still-alive group
 *        at once, so one composite kills the rest of its group without ever
 *        testing them. Within a round the runner sweeps witness by witness
 *        across all candidates, compacting survivors globally between
 *        witnesses — witness 2 only ever sees what survived witness 1.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_mr_gpu.cu, where it was the tail of a
 *               370-line main().
 * ───────────────────────────────────────────────────────────────────────────── */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <thread>
#include <cuda_runtime.h>
#include "driver/candidate.cuh"
#include "mr/miller_rabin_runner.cuh"
#include "perf/perf_node.cuh"
#include "tests/tests.cuh"
#include "bench/bench_ops.cuh"
#include "driver/input_parser.h"
#include "driver/cpu_runner.h"
#include "bench/compare_bench.h"
#include "driver/round_driver.h"
#include <chrono>

using hrc = std::chrono::high_resolution_clock;
static constexpr int BATCH_SIZE = MR_BATCH_SIZE;

void run_rounds(std::vector<GroupInfo> &groups, bool show_report, bool show_progress)
{
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


    auto t_global = hrc::now();
    std::vector<bool> alive(n_groups, true);

    for (int round = 0; round < max_rounds; round++)
    {
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

        std::vector<LazyCandidate> lcs(n_active);
        int batch_n_limbs = 0;
        for (int k = 0; k < n_active; k++)
        {
            int gi = active[k];
            lcs[k].equation  = groups[gi].equations[round];
            lcs[k].group_idx = gi;
            lcs[k].round_idx = round;
            int nl = lcs[k].natural_n_limbs();
            if (nl > batch_n_limbs) batch_n_limbs = nl;
        }

        std::vector<int> alive_idx(n_active);
        for (int i = 0; i < n_active; i++) alive_idx[i] = i;

        PerfCtrs round_perf;
        PerfNode round_tree{"TOTAL"};

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
                                                s, w, &round_perf,
                                                show_report ? &round_tree : nullptr,
                                                show_progress);

                for (int i = 0; i < sub_bsz; i++)
                    passed_this_witness[start + i] = sub_res[i];
            }

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

        if (show_report)
            print_perf_accumulated(round_perf, round_tree);

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
}
