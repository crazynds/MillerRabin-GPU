#pragma once
// ops/limb_storage.cuh — Limb storage abstraction for the modular pipeline.
//
// PROBLEM: the FFT-based multiplication backends (cuFFT / GPU-FFT / FFNT) compute
// in `double`, while the NTT backends compute in integers (`Data64`). Historically
// the modular layer (carry, sub, shift, reductions, runner) was written against
// `Data64`, forcing an int→double cast on every forward transform and a double→int
// cast on every inverse. Those casts are full extra memory passes per operation.
//
// SOLUTION: parametrize every limb-touching kernel on the *limb storage type*
// `LimbT`. For FFT backends LimbT == double, so the transform reads/writes its
// natural representation and the casts disappear; for NTT backends LimbT == Data64
// and the code is byte-identical to before.
//
// Numerical safety: all limb arithmetic is performed in `uint64_t` internally
// (mask / shift / borrow). Only the load/store at the kernel boundary crosses the
// double↔int line, and that is exact because:
//   • normalized limbs are < 2^LIMB_BITS  (≤ 2^16, exact in double),
//   • raw convolution coefficients are < 2^52 (guarded by each FFT ctor),
//   • double has a 52-bit mantissa → integers < 2^53 are represented exactly.
//
// limb_ld(double) replicates round_scatter's semantics (round to nearest, clamp
// negatives to 0) so a raw INTT coefficient read directly as double matches the
// value the old `round_scatter` kernel would have produced.

// NOTE: this header must be included AFTER the selected multiplication backend,
// so that `Data64` is already defined by the backend / GPU-NTT lib (it is
// `unsigned long` there — do NOT redefine it here).

#include "config.h"        // MUL_ALG identifiers (via constants.h)
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// ── Per-build selection of the limb storage type ───────────────────────────────
// One MUL_ALG per build ⇒ LimbT is a fixed typedef. Templated kernels are
// instantiated for this type; the typedef is what the reduction/runter layer uses
// to declare its buffers and pick the right instantiation.
#if MUL_ALG == MUL_FFT_CUFFT || MUL_ALG == MUL_FFT_GPUFFT || MUL_ALG == MUL_FFNT_GPUFFT
using LimbT = double;
#define LIMB_IS_REAL 1
#else
using LimbT = Data64;
#define LIMB_IS_REAL 0
#endif

// ── Load/store helpers (overloaded; templated kernels resolve the right one) ────

__host__ __device__ inline uint64_t limb_ld(Data64 x) { return (uint64_t)x; }

__host__ __device__ inline uint64_t limb_ld(double x)
{
    // Match round_scatter: round to nearest, clamp negatives to 0.
#ifdef __CUDA_ARCH__
    long long v = llround(x);
#else
    long long v = (long long)(x < 0.0 ? x - 0.5 : x + 0.5);
#endif
    return v < 0 ? 0ull : (uint64_t)v;
}

__host__ __device__ inline void limb_st(Data64 &dst, uint64_t v) { dst = (Data64)v; }
__host__ __device__ inline void limb_st(double &dst, uint64_t v) { dst = (double)v; }

// ── Host ↔ device limb transfer (converts uint64↔LimbT when LIMB_IS_REAL) ───────
// Returns the cudaError so callers can wrap with their CU() macro. `count` is the
// number of limbs (NOT bytes). When LimbT == Data64 these are plain cudaMemcpy.

inline cudaError_t limb_upload(LimbT *d_dst, const uint64_t *h_src, size_t count)
{
#if LIMB_IS_REAL
    std::vector<LimbT> tmp(count);
    for (size_t i = 0; i < count; i++)
        tmp[i] = (LimbT)h_src[i];
    return cudaMemcpy(d_dst, tmp.data(), count * sizeof(LimbT), cudaMemcpyHostToDevice);
#else
    return cudaMemcpy(d_dst, h_src, count * sizeof(LimbT), cudaMemcpyHostToDevice);
#endif
}

inline cudaError_t limb_download(uint64_t *h_dst, const LimbT *d_src, size_t count)
{
#if LIMB_IS_REAL
    std::vector<LimbT> tmp(count);
    cudaError_t e = cudaMemcpy(tmp.data(), d_src, count * sizeof(LimbT), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < count; i++)
        h_dst[i] = (uint64_t)tmp[i];
    return e;
#else
    return cudaMemcpy(h_dst, d_src, count * sizeof(LimbT), cudaMemcpyDeviceToHost);
#endif
}
