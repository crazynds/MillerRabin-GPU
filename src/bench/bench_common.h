/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_common.h
 * ROLE   stat records shared by the self-benchmark phases
 *
 * HOW    Plain structs accumulating counts and wall-clock time.
 *        Deliberately free of CUDA and GMP so both the GPU and the CPU side
 *        can include it without dragging in either toolchain.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <cstdio>
#include <algorithm>

// Accumulated result of a "run as many as possible in T seconds" phase.
struct ThroughputStats {
    long long count = 0;
    double elapsed_s = 0.0;

    double per_sec() const { return elapsed_s > 0.0 ? count / elapsed_s : 0.0; }
    double per_hour() const { return per_sec() * 3600.0; }
};

// Accumulated result of a "run N single-shot latency probes" phase.
struct LatencyStats {
    long long count = 0;
    double total_ms = 0.0;
    double min_ms = 1e18;
    double max_ms = 0.0;

    void add(double ms)
    {
        count++;
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    double avg_ms() const { return count > 0 ? total_ms / count : 0.0; }
};

inline void print_throughput_line(const char *label, const ThroughputStats &s)
{
    printf("  %-18s %10lld tests in %8.2fs   =>  %14.2f tests/h\n",
           label, s.count, s.elapsed_s, s.per_hour());
}

inline void print_latency_line(const char *label, const LatencyStats &s)
{
    printf("  %-18s %6lld runs   min %10.2f ms   avg %10.2f ms   max %10.2f ms\n",
           label, s.count, s.min_ms, s.avg_ms(), s.max_ms);
}
