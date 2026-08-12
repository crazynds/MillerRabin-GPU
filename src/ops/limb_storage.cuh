/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/limb_storage.cuh
 * ROLE   the limb storage type and its load/store boundary
 *
 * HOW    Every limb-touching kernel is templated on LimbT so it reads and
 *        writes the backend natural representation: double for the FFT
 *        backends, and for the NTT backends uint32 under MR_LIMB32,
 *        otherwise Data64. RawT stays 64-bit — raw INTT coefficients reach
 *        ~2^58 and are never narrowed. Only the carry layer crosses from
 *        RawT to LimbT.
 *
 * NOTE   All limb arithmetic happens in uint64 registers; the width only
 *        shows at the memory boundary. That is what makes the narrow
 *        storage safe: a normalized limb is below 2^LIMB_BITS by
 *        construction.
 *
 * CHANGELOG
 *   2026-08-11  Added the uint32 branch (MR_LIMB32) and split RawT out of
 *               LimbT. Measured -3.4% time and -47% peak VRAM at 100k
 *               digits.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "config.h"
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
#elif defined(MR_LIMB32)
// A normalized limb is < 2^LIMB_BITS <= 2^32, so storing it in 64 bits wastes
// half of every byte moved by the limb-side kernels (carry output, subtract,
// shift, copy) — and those run at 86-92% of peak DRAM bandwidth, i.e. they are
// bandwidth-bound and nothing but moving fewer bytes can speed them up.
// RawT stays 64-bit: the *spectral* and raw-INTT coefficients reach ~2^58.
using LimbT = uint32_t;
#define LIMB_IS_REAL 0
#else
using LimbT = Data64;
#define LIMB_IS_REAL 0
#endif

// Storage type of the raw (un-normalized) convolution coefficients produced by
// the inverse transform: the transform's own word type, which reaches ~2^58 and
// so is never narrowed. Only the carry layer crosses from RawT to LimbT.
#if LIMB_IS_REAL
using RawT = double;
#else
using RawT = Data64;
#endif

// ── Load/store helpers (overloaded; templated kernels resolve the right one) ────

__host__ __device__ inline uint64_t limb_ld(Data64 x) { return (uint64_t)x; }
#ifdef MR_LIMB32
__host__ __device__ inline uint64_t limb_ld(uint32_t x) { return (uint64_t)x; }
__host__ __device__ inline void limb_st(uint32_t &dst, uint64_t v) { dst = (uint32_t)v; }
#endif

__host__ __device__ inline uint64_t limb_ld(double x)
{
#ifdef __CUDA_ARCH__
    long long v = llround(x);
#else
    long long v = (long long)(x < 0.0 ? x - 0.5 : x + 0.5);
#endif
    return v < 0 ? 0ull : (uint64_t)v;
}

__host__ __device__ inline void limb_st(Data64 &dst, uint64_t v) { dst = (Data64)v; }
__host__ __device__ inline void limb_st(double &dst, uint64_t v) { dst = (double)v; }

// ── Host ↔ device limb transfer ────────────────────────────────────────────────
// The host always speaks uint64 limbs. Whenever LimbT is a different width or
// representation (double for the FFT backends, uint32 under MR_LIMB32) the
// transfer has to convert element-wise; otherwise it is a plain cudaMemcpy.
// Returns the cudaError so callers can wrap with their CU() macro. `count` is the
// number of limbs (NOT bytes).
#if LIMB_IS_REAL || defined(MR_LIMB32)
#define LIMB_NEEDS_CONVERT 1
#else
#define LIMB_NEEDS_CONVERT 0
#endif

inline cudaError_t limb_upload(LimbT *d_dst, const uint64_t *h_src, size_t count)
{
#if LIMB_NEEDS_CONVERT
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
#if LIMB_NEEDS_CONVERT
    std::vector<LimbT> tmp(count);
    cudaError_t e = cudaMemcpy(tmp.data(), d_src, count * sizeof(LimbT), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < count; i++)
        h_dst[i] = (uint64_t)tmp[i];
    return e;
#else
    return cudaMemcpy(h_dst, d_src, count * sizeof(LimbT), cudaMemcpyDeviceToHost);
#endif
}
