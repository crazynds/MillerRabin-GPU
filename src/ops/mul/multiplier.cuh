/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/mul/multiplier.cuh
 * ROLE   compile-time choice of the multiplication backend
 *
 * HOW    Resolves MUL_ALG to a single `Multiplier` type alias.
 *
 * NOTE   The reductions and the orchestrator are written against that alias
 *        and contain no per-backend #if, so switching algorithm is one line
 *        in params.cmake plus a rebuild.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "config.h"

#ifndef MUL_ALG
#error "MUL_ALG not defined (params.cmake → config.h). Use MUL_SCHOOLBOOK | MUL_MERGE_GPUNTT | MUL_4STEP_GPUNTT."
#endif

// Selection of the multiplication class. SCHOOLBOOK and NTT_MERGE share the
// "merge" class (BigIntNTTBatch) — the modular reduction always uses NTT, and the
// merge class provides both the NTT and the schoolbook_*. Only NTT_4STEP swaps the class.
#if MUL_ALG == MUL_4STEP_GPUNTT
#include "ops/mul/ntt_4step.cuh"
using Multiplier = Ntt4StepBatch;
#elif MUL_ALG == MUL_FFT_CUFFT
#include "ops/mul/fft_cufft.cuh"
using Multiplier = FftCuFFTBatch;
#elif MUL_ALG == MUL_FFT_GPUFFT
#include "ops/mul/fft_gpufft.cuh"
using Multiplier = FftGpuFftBatch;
#elif MUL_ALG == MUL_FFNT_GPUFFT
#include "ops/mul/fft_ffnt.cuh"
using Multiplier = FftFFNTBatch;
#elif MUL_ALG == MUL_MERGE_GPUNTT || MUL_ALG == MUL_SCHOOLBOOK
#include "ops/mul/ntt_merge.cuh"
using Multiplier = BigIntNTTBatch;
#else
#error "MUL_ALG invalid. Use MUL_SCHOOLBOOK | MUL_MERGE_GPUNTT | MUL_4STEP_GPUNTT | MUL_FFT_CUFFT."
#endif
