/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/util/cuda_check.cuh
 * ROLE   CU(expr) — turn a failing CUDA call into an exception
 *
 * HOW    Wraps a cudaError_t-returning call and throws std::runtime_error
 *        carrying the stringified expression and cudaGetErrorString. Errors
 *        surface at the call site instead of silently corrupting later
 *        work.
 *
 * NOTE   The macro was copy-pasted into eight translation units before
 *        this; keeping one definition means one place to change how
 *        failures are reported.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from the eight files that each defined their own
 *               copy.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifndef CU
#define CU(expr) \
    do \
    { \
        cudaError_t _e = (expr); \
        if (_e != cudaSuccess) \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)
#endif
