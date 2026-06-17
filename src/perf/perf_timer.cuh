// perf/perf_timer.cuh — GPU stopwatch based on a ring of CUDA events.
//
// start()/stop(node) record markers on the stream WITHOUT synchronizing; flush() synchronizes
// ONCE and accumulates each interval into the corresponding PerfNode. Same scheme as the
// old TSTART/TSTOP, but the target is a PerfNode* (dynamic graph) instead of a
// fixed struct field.
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "perf/perf_node.cuh"

class PerfTimer
{
public:
    static constexpr int RING = 64; // markers per public call (slack for long pipelines)

    void init()
    {
        for (int i = 0; i <= RING; i++)
            check(cudaEventCreate(&ev_[i]));
    }
    void destroy()
    {
        for (int i = 0; i <= RING; i++)
            if (ev_[i])
                cudaEventDestroy(ev_[i]);
    }

    // Start marker of the next section. The perf gate lives in the TSTART macro
    // (checks BatchModCtx::perf_enabled); here we only record the event.
    void start(cudaStream_t s)
    {
        check(cudaEventRecord(ev_[cur_], s));
    }
    // End marker; records the target node. No sync.
    void stop(PerfNode *node, cudaStream_t s)
    {
        check(cudaEventRecord(ev_[cur_ + 1], s));
        acc_[cur_] = node;
        node->calls++;
        cur_++;
    }
    // Synchronizes the last marker and accumulates all pending intervals.
    // Safe to call always: if nothing was recorded (cur_ == 0), it is a no-op.
    void flush(cudaStream_t)
    {
        if (cur_ == 0)
            return;
        check(cudaEventSynchronize(ev_[cur_]));
        for (int i = 0; i < cur_; i++)
        {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, ev_[i], ev_[i + 1]));
            acc_[i]->ms += ms;
        }
        cur_ = 0;
    }

private:
    static void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("[perf] CUDA: ") + cudaGetErrorString(e));
    }
    cudaEvent_t ev_[RING + 1] = {};
    PerfNode *acc_[RING] = {};
    int cur_ = 0;
};
