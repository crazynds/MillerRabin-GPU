// perf/perf_node.cuh — Profile tree built dynamically at runtime.
//
// Each group/operation calls branch() to create children with pre-declared names,
// accessed later by index (child(int)). No fixed fields per algorithm.
// The report simply walks the tree — it does not depend on any external struct.
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <initializer_list>
#include <cstdio>
#include "helpers/time_format.h"

class PerfNode
{
public:
    std::string name;
    double ms = 0.0;       // own time (timed leaves)
    long long calls = 0;   // number of timed calls
    std::string note;      // optional annotation (e.g.: bandwidth GB/s)
    std::vector<std::unique_ptr<PerfNode>> children;

    explicit PerfNode(std::string n) : name(std::move(n)) {}

    // Creates and attaches a child (optionally with pre-declared sub-children).
    // The returned pointer is stable (stored in a unique_ptr).
    PerfNode *branch(const std::string &n,
                     std::initializer_list<std::string> kids = {})
    {
        children.push_back(std::make_unique<PerfNode>(n));
        PerfNode *p = children.back().get();
        for (auto &k : kids)
            p->children.push_back(std::make_unique<PerfNode>(k));
        return p;
    }

    // Access by index (stable after the tree is built).
    PerfNode *child(int i) { return children[(size_t)i].get(); }
    const PerfNode *child(int i) const { return children[(size_t)i].get(); }

    // Displayed time: own if leaf; sum of children if group.
    double total_ms() const
    {
        if (children.empty())
            return ms;
        double t = 0.0;
        for (auto &c : children)
            t += c->total_ms();
        return t;
    }

    bool is_leaf() const { return children.empty(); }
};

// ── Tree printing ─────────────────────────────────────────────────────────────
namespace perf_detail
{
    // Visible width (UTF-8 code points) to align labels with ├─│.
    inline std::string padlbl(const std::string &s, int w)
    {
        int cols = 0;
        for (unsigned char c : s)
            if ((c & 0xC0) != 0x80)
                cols++;
        std::string r = s;
        for (int i = cols; i < w; i++)
            r += ' ';
        return r;
    }

    inline void rec(const PerfNode *n, const std::string &prefix, bool last,
                    double root_ms, int lbl_w)
    {
        std::string label = padlbl(prefix + (last ? "└─ " : "├─ ") + n->name, lbl_w);
        double ms = n->total_ms();
        double pct = root_ms > 0 ? ms * 100.0 / root_ms : 0.0;
        const char *note = n->note.empty() ? "" : n->note.c_str();
        if (n->is_leaf())
            printf("  %s %12s  %5.1f%%  %12s/call %s\n",
                   label.c_str(), fmt_time_ms((float)ms).c_str(), pct,
                   fmt_time_ms((float)(n->calls > 0 ? ms / n->calls : 0.0)).c_str(), note);
        else
            printf("  %s %12s  %5.1f%% %s\n",
                   label.c_str(), fmt_time_ms((float)ms).c_str(), pct, note);

        // Sorts children by decreasing time.
        std::vector<const PerfNode *> ks;
        for (auto &k : n->children)
            ks.push_back(k.get());
        for (size_t i = 0; i + 1 < ks.size(); i++)
            for (size_t j = 0; j + 1 < ks.size() - i; j++)
                if (ks[j]->total_ms() < ks[j + 1]->total_ms())
                    std::swap(ks[j], ks[j + 1]);
        std::string cp = prefix + (last ? "   " : "│  ");
        for (size_t i = 0; i < ks.size(); i++)
            rec(ks[i], cp, i + 1 == ks.size(), root_ms, lbl_w);
    }
} // namespace perf_detail

// Prints the entire tree starting from `root` (root = 100%).
inline void print_perf_tree(const PerfNode &root, int lbl_w = 36)
{
    double root_ms = root.total_ms();
    printf("  %s %12s  %5.1f%%\n",
           perf_detail::padlbl(root.name, lbl_w).c_str(),
           fmt_time_ms((float)root_ms).c_str(), 100.0);
    std::vector<const PerfNode *> ks;
    for (auto &k : root.children)
        ks.push_back(k.get());
    for (size_t i = 0; i + 1 < ks.size(); i++)
        for (size_t j = 0; j + 1 < ks.size() - i; j++)
            if (ks[j]->total_ms() < ks[j + 1]->total_ms())
                std::swap(ks[j], ks[j + 1]);
    for (size_t i = 0; i < ks.size(); i++)
        perf_detail::rec(ks[i], "", i + 1 == ks.size(), root_ms, lbl_w);
}
