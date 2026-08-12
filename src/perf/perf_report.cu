/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/perf/perf_report.cu
 * ROLE   human-readable timing reports for a Miller-Rabin run
 *
 * HOW    Formats the PerfCtrs counters and the PerfNode tree into the
 *        tables printed after a run. merge_perf_tree folds the per-sub-
 *        batch trees into one before printing.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from mr/mr_internals.cu; it is reporting, not MR
 *               internals.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "mr/mr_internals.cuh"
#include "ops/mul/multiplier.cuh"
#include "util/cuda_check.cuh"
#include "util/time_format.h"
#include <cstdio>
#include <vector>

// ── Prints performance report ─────────────────────────────────────────

void print_perf_simple(const PerfCtrs &perf)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms = window_ms + perf.check_ms + perf.setup_ms + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v)
    { return total_ms > 0 ? v * 100.0f / total_ms : 0.0f; };

    double memcpy_gb = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0 ? memcpy_gb / (perf.memcpy_ms / 1000.0) : 0.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Time profile — WINDOW_BITS=%-2d                              ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n", fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms / perf.sq_calls : 0.0).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld win, %s/win)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms / perf.mul_calls : 0.0).c_str());
    printf("  table pre-compute       %12s  %5.1f%%\n", fmt_time_ms(perf.table_ms).c_str(), pct(perf.table_ms));
    printf("  CPU setup (to_mont)     %12s  %5.1f%%\n", fmt_time_ms(perf.setup_ms).c_str(), pct(perf.setup_ms));
    printf("  check                   %12s  %5.1f%%\n", fmt_time_ms(perf.check_ms).c_str(), pct(perf.check_ms));
    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    printf("  memcpy setup            %12s  %5.1f%%  %s\n", fmt_time_ms(perf.memcpy_ms).c_str(), pct(perf.memcpy_ms), gbps);
    printf("  ──────────────────────  %12s\n", fmt_time_ms(total_ms).c_str());
    carry_stats_print_and_reset();
}

void print_perf(const PerfCtrs &perf, BatchModCtx &mont)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms = window_ms + perf.check_ms + perf.setup_ms + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v)
    { return total_ms > 0 ? v * 100.0f / total_ms : 0.0f; };

    double memcpy_gb = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0 ? memcpy_gb / (perf.memcpy_ms / 1000.0) : 0.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Time profile — WINDOW_BITS=%-2d                              ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n", fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms / perf.sq_calls : 0.0).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld win, %s/win)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms / perf.mul_calls : 0.0).c_str());

    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    std::vector<BatchModCtx::HostPhase> host = {
        {"table pre-compute", perf.table_ms, ""},
        {"CPU setup (to_mont)", perf.setup_ms, ""},
        {"check + memcpy", perf.check_ms, ""},
        {"memcpy setup", perf.memcpy_ms, gbps},
    };
    mont.print_perf(total_ms, host);
    carry_stats_print_and_reset();
}

// ── merge_perf_tree ───────────────────────────────────────────────────────────
// Merges src into dst: adds ms/calls for matching nodes (by position).
// On first call (dst.children is empty) the tree structure is cloned from src.

void merge_perf_tree(PerfNode &dst, const PerfNode &src)
{
    dst.ms += src.ms;
    dst.calls += src.calls;
    if (src.children.empty())
        return;

    if (dst.children.empty())
    {
        for (auto &c : src.children)
        {
            dst.children.push_back(std::make_unique<PerfNode>(c->name));
            dst.children.back()->note = c->note;
            merge_perf_tree(*dst.children.back(), *c);
        }
    }
    else
    {
        for (size_t i = 0; i < src.children.size() && i < dst.children.size(); i++)
            merge_perf_tree(*dst.children[i], *src.children[i]);
    }
}

// ── print_perf_accumulated ────────────────────────────────────────────────────
// Prints the full timing report: GPU kernel tree (accumulated across all
// sub-batches/witnesses) annotated with host phases from PerfCtrs.

void print_perf_accumulated(const PerfCtrs &perf, PerfNode &tree)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms = window_ms + perf.check_ms + perf.setup_ms + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v)
    { return total_ms > 0 ? v * 100.0f / total_ms : 0.0f; };

    double memcpy_gb = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0
                             ? memcpy_gb / (perf.memcpy_ms / 1000.0)
                             : 0.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Time profile — WINDOW_BITS=%-2d                              ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n",
           fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms / perf.sq_calls : 0.0f).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld win, %s/win)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms / perf.mul_calls : 0.0f).c_str());
    printf("  table pre-compute       %12s  %5.1f%%\n",
           fmt_time_ms(perf.table_ms).c_str(), pct(perf.table_ms));
    printf("  CPU setup (to_mont)     %12s  %5.1f%%\n",
           fmt_time_ms(perf.setup_ms).c_str(), pct(perf.setup_ms));
    printf("  check                   %12s  %5.1f%%\n",
           fmt_time_ms(perf.check_ms).c_str(), pct(perf.check_ms));
    {
        char gbps[32];
        snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
        printf("  memcpy setup            %12s  %5.1f%%  %s\n",
               fmt_time_ms(perf.memcpy_ms).c_str(), pct(perf.memcpy_ms), gbps);
    }
    printf("  ──────────────────────  %12s\n", fmt_time_ms(total_ms).c_str());

    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    std::vector<BatchModCtx::HostPhase> host = {
        {"table pre-compute", perf.table_ms, ""},
        {"CPU setup (to_mont)", perf.setup_ms, ""},
        {"check", perf.check_ms, ""},
        {"memcpy setup", perf.memcpy_ms, gbps},
    };
    if (!host.empty())
    {
        PerfNode *h = tree.branch("setup / host");
        for (auto &hp : host)
        {
            PerfNode *leaf = h->branch(hp.name);
            leaf->ms = hp.ms;
            leaf->calls = 1;
            leaf->note = hp.note;
        }
    }

    if (total_ms > 0.0)
    {
        double gpu_tree_ms = 0.0;
        for (auto &c : tree.children)
            gpu_tree_ms += c->total_ms();
        double others = (double)total_ms - gpu_tree_ms;
        if (others > 0.5)
        {
            PerfNode *o = tree.branch("others (overhead)");
            o->ms = others;
            o->calls = 1;
        }
    }

    printf("\n  Application breakdown (accumulated across all sub-batches):\n");
    print_perf_tree(tree);

    auto collect_leaves = [](const PerfNode &n, std::vector<std::pair<std::string, double>> &out)
    {
        std::function<void(const PerfNode &)> walk = [&](const PerfNode &nd)
        {
            if (nd.children.empty())
            {
                out.push_back({nd.name, nd.ms});
                return;
            }
            for (auto &c : nd.children)
                walk(*c);
        };
        walk(n);
    };
    auto has = [](const std::string &s, const char *sub)
    {
        return s.find(sub) != std::string::npos;
    };

    std::vector<std::pair<std::string, double>> leaves;
    for (auto &c : tree.children)
        if (c->name == "mul" || c->name == "sq")
            collect_leaves(*c, leaves);

    double ntt_t = 0, pw_t = 0, carry_t = 0, shift_t = 0, vadd_t = 0, sub_t = 0, copy_t = 0, cs_t = 0;
    for (auto &lv : leaves)
    {
        const std::string &nm = lv.first;
        double ms = lv.second;
        if (has(nm, "carry"))
            carry_t += ms;
        else if (has(nm, "pmul") || has(nm, "psq"))
            pw_t += ms;
        else if (has(nm, "ntt"))
            ntt_t += ms;
        else if (has(nm, "vadd"))
            vadd_t += ms;
        else if (has(nm, "shift"))
            shift_t += ms;
        else if (has(nm, "copy"))
            copy_t += ms;
        else if (has(nm, "cond_sub"))
            cs_t += ms;
        else if (has(nm, "sub"))
            sub_t += ms;
    }
    double tree_total = tree.total_ms();
    auto tp = [&](double v)
    { return tree_total > 0 ? v * 100.0 / tree_total : 0.0; };
    auto crow = [&](const char *name, double ms)
    {
        if (ms > 0)
            printf("     %-22s %12s  %5.1f%%\n", name,
                   fmt_time_ms((float)ms).c_str(), tp(ms));
    };
    printf("\n  by kernel type (accumulated):\n");
    crow("NTT/INTT", ntt_t);
    crow("pointwise (pmul)", pw_t);
    crow("carry", carry_t);
    crow("shift", shift_t);
    crow("sum (vadd)", vadd_t);
    crow("sub (T-qn)", sub_t);
    crow("cond_sub", cs_t);
    crow("copy_out", copy_t);

    carry_stats_print_and_reset();
}
