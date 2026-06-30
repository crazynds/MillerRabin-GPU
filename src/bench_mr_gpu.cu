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
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>

#include "candidate.cuh"
#include "miller_rabin_runner.cuh"
#include "correctness_tests.cuh"
#include "helpers/bench_ops.cuh"

using hrc = std::chrono::high_resolution_clock;

static constexpr int BATCH_SIZE = MR_BATCH_SIZE;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Packs flat arrays [bsz * n_limbs] from a list of NumberCandidate pointers.

// Tests a batch of NumberCandidates on the GPU with the chosen witnesses.
// Picks the s=1 fast path when all candidates in the batch share s==1.
static std::vector<bool> test_batch(
    const std::vector<NumberCandidate *> &cands,
    int n_limbs,
    const char *label,
    bool show_report,
    bool show_progress)
{
    std::vector<uint64_t> N_batch, Nm1_batch, d_batch;
    pack_batch(cands, n_limbs, N_batch, Nm1_batch, d_batch);

    int bsz = (int)cands.size();
    int s = cands[0]->s;

    BatchModCtx ctx(N_batch, n_limbs, bsz);
    if (s == 1)
        return gpu_miller_rabin_s1(ctx, d_batch, Nm1_batch, bsz, DEFAULT_WITNESSES,
                                   label, show_report, show_progress);
    else
        return gpu_miller_rabin(ctx, d_batch, Nm1_batch, s, bsz, DEFAULT_WITNESSES,
                                label, show_report, show_progress);
}

// ── Input parsing ─────────────────────────────────────────────────────────────

// Parses the input file and returns a flat list of GroupCandidates (not yet built).
// Line format: "[group_id:]equation"  (#-comments and blank lines ignored).
// Lines without ':' are treated as singleton groups with an empty label — each
// gets its own unique auto-label so it never merges with other lines.
static std::vector<GroupCandidate> parse_input(const char *path)
{
    std::ifstream fin(path);
    if (!fin)
        throw std::runtime_error(std::string("Cannot open file: ") + path);

    // Preserve insertion order while grouping.
    std::vector<std::string> order;               // ordered unique labels
    std::map<std::string, GroupCandidate> groups; // label → group
    int auto_id = 0;

    std::string line;
    while (std::getline(fin, line))
    {
        // Strip trailing whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;

        std::string label, equation;
        auto colon = line.find(':');
        if (colon != std::string::npos)
        {
            label = line.substr(0, colon);
            equation = line.substr(colon + 1);
        }
        else
        {
            label = "__auto_" + std::to_string(auto_id++);
            equation = line;
        }

        // Trim leading/trailing whitespace from both parts
        auto trim = [](std::string &s)
        {
            while (!s.empty() && std::isspace((unsigned char)s.front()))
                s.erase(s.begin());
            while (!s.empty() && std::isspace((unsigned char)s.back()))
                s.pop_back();
        };
        trim(label);
        trim(equation);
        if (equation.empty())
            continue;

        if (groups.find(label) == groups.end())
        {
            order.push_back(label);
            groups[label].label = label;
        }
        groups[label].equations.push_back(equation);
    }

    std::vector<GroupCandidate> result;
    result.reserve(order.size());
    for (auto &lbl : order)
        result.push_back(std::move(groups[lbl]));
    return result;
}

// ── CPU Miller-Rabin (via GMP) ────────────────────────────────────────────────

// Evaluates the equation and runs Miller-Rabin on CPU via GMP.
// Returns {is_prime, elapsed_ms}.
static std::pair<bool, double> cpu_test_equation(const std::string &equation)
{
    mpz_t N;
    mpz_init(N);
    EquationParser::eval(equation, N);
    auto t0 = hrc::now();
    bool result = mpz_probab_prime_p(N, (int)DEFAULT_WITNESSES.size()) > 0;
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    hrc::now() - t0)
                    .count() /
                1000.0;
    mpz_clear(N);
    return {result, ms};
}

// Runs the round-based group testing on CPU (one candidate at a time, no batch).
static void run_cpu_mode(
    std::vector<GroupCandidate> &groups,
    bool show_report,
    bool show_progress,
    bool cpu_parallel)
{
    int n_groups = (int)groups.size();
    int max_rounds = 0;
    for (auto &g : groups)
        if ((int)g.equations.size() > max_rounds)
            max_rounds = (int)g.equations.size();

    int n_threads = cpu_parallel
                        ? (int)std::thread::hardware_concurrency()
                        : 1;
    if (n_threads < 1)
        n_threads = 1;

    {
        int total_eqs = 0;
        for (auto &g : groups)
            total_eqs += (int)g.equations.size();
        printf("Groups: %d  total equations: %d  max rounds: %d\n",
               n_groups, total_eqs, max_rounds);
        printf("CPU mode — Miller-Rabin (GMP)  witnesses: %d  threads: %d\n\n",
               (int)DEFAULT_WITNESSES.size(), n_threads);
    }

    auto t_global = hrc::now();
    std::vector<bool> alive(n_groups, true);

    // Global stats across all rounds for the final report
    double global_t_min = 1e18, global_t_max = 0.0, global_t_sum = 0.0;
    int global_n_tested = 0;

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

        double t_min = 1e18, t_max = 0.0, t_sum = 0.0;
        int n_tested = 0;
        int survivors_this_round = 0;

        std::mutex print_mtx;
        std::mutex stats_mtx;
        std::atomic<int> work_idx{0};
        // results: 1=prime, 0=composite, -1=not yet
        std::vector<int> results(n_active, -1);

        auto worker = [&]()
        {
            while (true)
            {
                int idx = work_idx.fetch_add(1);
                if (idx >= n_active)
                    break;

                int gi = active[idx];
                const std::string &eq = groups[gi].equations[round];
                const std::string &lbl = groups[gi].label;

                if (show_progress)
                {
                    std::lock_guard<std::mutex> lk(print_mtx);
                    printf("  [%d/%d] core %s testing [%s]: %s\n",
                           idx + 1, n_active,
                           cpu_parallel ? std::to_string(
                                              (int)(std::hash<std::thread::id>{}(
                                                        std::this_thread::get_id()) %
                                                    (unsigned)n_threads))
                                              .c_str()
                                        : "0",
                           lbl.empty() || lbl.rfind("__auto_", 0) == 0 ? "#" : lbl.c_str(),
                           eq.c_str());
                    fflush(stdout);
                }

                auto [probably_prime, ms] = cpu_test_equation(eq);
                results[idx] = probably_prime ? 1 : 0;

                {
                    std::lock_guard<std::mutex> lk(stats_mtx);
                    t_sum += ms;
                    if (ms < t_min) t_min = ms;
                    if (ms > t_max) t_max = ms;
                    n_tested++;
                }

                if (show_report)
                {
                    std::lock_guard<std::mutex> lk(print_mtx);
                    printf("  [%d/%d] [%s] %7.1f ms  %s  %s\n",
                           idx + 1, n_active,
                           lbl.empty() || lbl.rfind("__auto_", 0) == 0 ? "#" : lbl.c_str(),
                           ms,
                           probably_prime ? "PROBABLY PRIME" : "composite     ",
                           eq.c_str());
                    fflush(stdout);
                }
            }
        };

        if (n_threads > 1)
        {
            std::vector<std::thread> threads;
            threads.reserve(n_threads);
            for (int t = 0; t < n_threads; t++)
                threads.emplace_back(worker);
            for (auto &th : threads)
                th.join();
        }
        else
        {
            worker();
        }

        // Apply results
        for (int idx = 0; idx < n_active; idx++)
        {
            if (results[idx] == 1)
                survivors_this_round++;
            else
                alive[active[idx]] = false;
        }

        // Accumulate global stats
        global_t_sum += t_sum;
        global_n_tested += n_tested;
        if (t_min < global_t_min) global_t_min = t_min;
        if (t_max > global_t_max) global_t_max = t_max;

        double t_r = std::chrono::duration_cast<std::chrono::milliseconds>(
                         hrc::now() - t_round)
                         .count() /
                     1000.0;
        int eliminated = n_active - survivors_this_round;
        double t_avg = n_tested > 0 ? t_sum / n_tested : 0.0;
        printf("Round %d: %.2fs  —  %d survived, %d eliminated"
               "  |  min %.1fms  avg %.1fms  max %.1fms\n",
               round + 1, t_r, survivors_this_round, eliminated,
               t_min, t_avg, t_max);
    }

    double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(
                         hrc::now() - t_global)
                         .count() /
                     1000.0;

    printf("\n=== Results ===\n");
    int n_winners = 0;
    for (int gi = 0; gi < n_groups; gi++)
    {
        if (alive[gi])
        {
            n_winners++;
            auto &g = groups[gi];
            if (g.label.empty() || g.label.rfind("__auto_", 0) == 0)
            {
                printf("  PRIME: %s\n", g.equations[0].c_str());
            }
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

    printf("Total time: %.2fs\n", total_s);

    if (show_report && global_n_tested > 0)
    {
        double g_avg = global_t_sum / global_n_tested;
        printf("\n=== CPU Report ===\n");
        printf("  Iterations : %d\n", global_n_tested);
        printf("  Total time : %.2fs\n", total_s);
        printf("  Avg / iter : %.2f ms\n", g_avg);
        printf("  Min / iter : %.2f ms\n", global_t_min);
        printf("  Max / iter : %.2f ms\n", global_t_max);
    }
}

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
    bool cpu_parallel = false;
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
            cpu_parallel = true;
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
                " [--bench-ops] [--bench-ops-long] [--cpu] [--cpu-parallel] <input.txt>\n",
                argv[0]);
        return 1;
    }

    // ── Load & build candidates ───────────────────────────────────────────────
    std::vector<GroupCandidate> groups;
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

    // ── CPU mode: GMP Miller-Rabin, no GPU, no batching ──────────────────────
    if (cpu_mode)
    {
        run_cpu_mode(groups, show_report, show_progress, cpu_parallel);
        return 0;
    }

    int n_groups = (int)groups.size();
    int max_rounds = 0;
    for (auto &g : groups)
        if ((int)g.equations.size() > max_rounds)
            max_rounds = (int)g.equations.size();

    // Compute n_limbs per group (may differ if equations produce different-sized numbers)
    printf("Building candidates (GMP)...\n");
    fflush(stdout);
    for (auto &g : groups)
    {
        try
        {
            g.build();
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "Error building group \"%s\": %s\n", g.label.c_str(), e.what());
            return 1;
        }
    }

    // For batch compatibility within a round: groups sharing the same n_limbs
    // can be co-batched. We collect n_limbs per group so batching is correct.
    {
        // Just a summary print — actual per-group n_limbs used in batching below.
        int total_eqs = 0;
        for (auto &g : groups)
            total_eqs += (int)g.equations.size();
        printf("Groups: %d  total equations: %d  max rounds: %d\n",
               n_groups, total_eqs, max_rounds);
        printf("Batch size: %d  witnesses: %d\n\n", BATCH_SIZE, (int)DEFAULT_WITNESSES.size());
    }

    // Correctness tests (uses the first batch of the first group)
    if (run_tests)
    {
        auto &g0 = groups[0];
        int bsz_test = std::min((int)g0.cands.size(), BATCH_SIZE);
        std::vector<NumberCandidate *> test_cands;
        for (int i = 0; i < bsz_test; i++)
            test_cands.push_back(&g0.cands[i]);
        std::vector<uint64_t> N_test, Nm1_test, d_test;
        pack_batch(test_cands, g0.n_limbs, N_test, Nm1_test, d_test);
        BatchModCtx ctx_test(N_test, g0.n_limbs, bsz_test);
        run_correctness_tests(ctx_test, N_test);
        run_known_prime_tests();
        run_general_s_prime_tests();
        run_s1_nextprime_tests();
    }

    // ── Round-based group testing ─────────────────────────────────────────────
    //
    // alive[i] = true  → group i is still in the running.
    // In each round, for every still-alive group we dispatch the next equation.
    // Groups are batched by n_limbs so that the GPU context is homogeneous.
    // If any equation in a group is composite, the group is eliminated.

    auto t_global = hrc::now();
    std::vector<bool> alive(n_groups, true);

    for (int round = 0; round < max_rounds; round++)
    {
        // Collect groups that still have a candidate for this round
        std::vector<int> active;
        for (int gi = 0; gi < n_groups; gi++)
            if (alive[gi] && round < (int)groups[gi].cands.size())
                active.push_back(gi);

        if (active.empty())
            break;

        // Sort active groups by n_limbs (descending) so that candidates of the same
        // size cluster together within each BATCH_SIZE chunk.  When a chunk contains
        // groups of different sizes the batch is normalised to the largest n_limbs in
        // that chunk: smaller candidates are zero-padded, which is safe because
        // pack_batch zero-initialises the output buffer before copying the limbs.
        std::sort(active.begin(), active.end(), [&](int a, int b)
                  { return groups[a].n_limbs > groups[b].n_limbs; });

        printf("\n=== Round %d (%d groups active) ===\n", round + 1, (int)active.size());
        fflush(stdout);
        auto t_round = hrc::now();
        int survivors_this_round = 0;

        int total = (int)active.size();
        int n_batches = (total + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int b = 0; b < n_batches; b++)
        {
            int bstart = b * BATCH_SIZE;
            int bend = std::min(bstart + BATCH_SIZE, total);
            int bsz = bend - bstart;

            // Determine the widest n_limbs in this chunk — all others are zero-padded.
            int batch_n_limbs = 0;
            for (int k = bstart; k < bend; k++)
                if (groups[active[k]].n_limbs > batch_n_limbs)
                    batch_n_limbs = groups[active[k]].n_limbs;

            std::vector<NumberCandidate *> batch_cands;
            std::vector<int> batch_gidx;
            for (int k = bstart; k < bend; k++)
            {
                int gi = active[k];
                batch_cands.push_back(&groups[gi].cands[round]);
                batch_gidx.push_back(gi);
            }

            char label_buf[64];
            snprintf(label_buf, sizeof(label_buf), "r%d b%d n=%d",
                     round + 1, b + 1, batch_n_limbs);
            auto results = test_batch(batch_cands, batch_n_limbs, label_buf,
                                      show_report, show_progress);

            for (int k = 0; k < bsz; k++)
            {
                int gi = batch_gidx[k];
                if (!results[k])
                {
                    alive[gi] = false; // composite — eliminate group
                }
                else
                {
                    survivors_this_round++;
                }
            }
        }

        double t_r = std::chrono::duration_cast<std::chrono::milliseconds>(
                         hrc::now() - t_round)
                         .count() /
                     1000.0;
        int eliminated = (int)active.size() - survivors_this_round;
        printf("Round %d: %.2fs  —  %d survived, %d eliminated\n",
               round + 1, t_r, survivors_this_round, eliminated);
    }

    double total = std::chrono::duration_cast<std::chrono::milliseconds>(
                       hrc::now() - t_global)
                       .count() /
                   1000.0;

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
            {
                // Singleton group without an explicit label
                printf("  PRIME: %s\n", g.equations[0].c_str());
            }
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
