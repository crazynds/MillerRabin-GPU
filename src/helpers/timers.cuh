#pragma once
// helpers/timers.cuh — internal macros shared by the modular context .cu files
// (batch_mod_ctx.cu, reductions/*.cu). Do NOT include in the public header: these macros
// reference members of BatchModCtx (timer, perf_enabled) and the stream variable `s`.

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// CUDA error check → exception.
#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// Start marker of the next section (no sync). No-op if perf_enabled == false.
#define TSTART()          \
    do                    \
    {                     \
        if (perf_enabled) \
            timer.start(s); \
    } while (0)

// End marker accumulating into PerfNode* `node` (no sync). timer.flush() synchronizes
// a single time at the end of the public function. No-op if perf_enabled == false.
#define TSTOP(node)            \
    do                         \
    {                          \
        if (perf_enabled)      \
            timer.stop((node), s); \
    } while (0)
