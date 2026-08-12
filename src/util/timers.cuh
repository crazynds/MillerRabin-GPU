/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/util/timers.cuh
 * ROLE   TSTART/TSTOP macros for the modular context
 *
 * HOW    Record CUDA events into the perf tree, compiling to nothing when
 *        perf_enabled is false.
 *
 * NOTE   Internal on purpose: the macros reference BatchModCtx members
 *        (timer, perf_enabled) and a stream named `s` in the enclosing
 *        scope, so they only work inside its .cu files. Do not include from
 *        a public header.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "util/cuda_check.cuh"

// CUDA error check → exception.

// Start marker of the next section (no sync). No-op if perf_enabled == false.
#define TSTART() \
    do \
    { \
        if (perf_enabled) \
            timer.start(s); \
    } while (0)

// End marker accumulating into PerfNode* `node` (no sync). timer.flush() synchronizes
// a single time at the end of the public function. No-op if perf_enabled == false.
#define TSTOP(node) \
    do \
    { \
        if (perf_enabled) \
            timer.stop((node), s); \
    } while (0)
