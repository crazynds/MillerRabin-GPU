// helpers/mod_perf.cu — Profile printing. Walks the dynamic graph (PerfNode)
// built in build_perf_nodes; no fixed fields nor #if per reduction algorithm.

#include "batch_mod_ctx.cuh"
#include "helpers/time_format.h"
#include <cstdio>
#include <string>
#include <vector>
#include <functional>

namespace
{
    // Recursive sum of the time of a subtree.
    double subtree_ms(const PerfNode &n)
    {
        if (n.children.empty())
            return n.ms;
        double t = 0;
        for (auto &c : n.children)
            t += subtree_ms(*c);
        return t;
    }

    // Collects (name, ms) from all the LEAVES of the tree.
    void collect_leaves(const PerfNode &n, std::vector<std::pair<std::string, double>> &out)
    {
        if (n.children.empty())
        {
            out.push_back({n.name, n.ms});
            return;
        }
        for (auto &c : n.children)
            collect_leaves(*c, out);
    }

    bool has(const std::string &s, const char *sub) { return s.find(sub) != std::string::npos; }
}

void BatchModCtx::print_perf(double app_total_ms, const std::vector<HostPhase> &host)
{
    // Attaches the host phases as leaves under "setup / host".
    if (!host.empty())
    {
        PerfNode *h = perf_root.branch("setup / host");
        for (auto &hp : host)
        {
            PerfNode *leaf = h->branch(hp.name);
            leaf->ms = hp.ms;
            leaf->calls = 1;
            leaf->note = hp.note;
        }
    }

    // "others (overhead)" = application total − everything measured in the tree.
    if (app_total_ms > 0.0)
    {
        double measured = 0;
        for (auto &c : perf_root.children)
            measured += subtree_ms(*c);
        double others = app_total_ms - measured;
        if (others > 0.0)
        {
            PerfNode *o = perf_root.branch("others (overhead)");
            o->ms = others;
            o->calls = 1;
        }
    }

    printf("\n  Application breakdown (unified tree):\n");
    print_perf_tree(perf_root);

    // ── Cross-cutting view by kernel type (accumulated over all leaves) ──
    std::vector<std::pair<std::string, double>> leaves;
    for (auto &c : perf_root.children)
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
        else if (has(nm, "ntt")) // catches ntt and intt
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
    double total = subtree_ms(perf_root);
    auto pct = [&](double v) { return total > 0 ? v * 100.0 / total : 0.0; };
    auto crow = [&](const char *name, double ms)
    {
        if (ms > 0)
            printf("     %-22s %12s  %5.1f%%\n", name, fmt_time_ms((float)ms).c_str(), pct(ms));
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
}
