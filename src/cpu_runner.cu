#include "cpu_runner.h"
#include "miller_rabin_runner.cuh"
#include "equation.h"
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <pthread.h>
#include <sched.h>
#include <cstdio>
#include <string>
#include <vector>

using hrc = std::chrono::high_resolution_clock;

std::pair<bool, double> cpu_test_equation(const std::string &equation)
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

void run_cpu_mode(
    std::vector<GroupInfo> &groups,
    bool show_report,
    bool show_progress,
    int n_threads)          // 1 = serial; >1 = parallel with core affinity
{
    int n_groups = (int)groups.size();
    int max_rounds = 0;
    for (auto &g : groups)
        if ((int)g.equations.size() > max_rounds)
            max_rounds = (int)g.equations.size();

    if (n_threads < 1) n_threads = 1;

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
                           n_threads > 1 ? std::to_string(
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
            int n_cores = (int)std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            threads.reserve(n_threads);
            for (int t = 0; t < n_threads; t++)
            {
                threads.emplace_back(worker);
                // Bind each thread to a distinct core (wraps around if n_threads > n_cores).
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(t % n_cores, &cpuset);
                pthread_setaffinity_np(threads.back().native_handle(),
                                       sizeof(cpu_set_t), &cpuset);
            }
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
