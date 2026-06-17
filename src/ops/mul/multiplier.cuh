// ops/mul/multiplier.cuh — COMPILE-TIME selection of the big-int multiplication backend.
//
// The reductions (Barrett/Montgomery) and the orchestrator are written against the
// `Multiplier` type and NEVER contain per-backend #if. Switching algorithm = switch
// MUL_ALG in params.cmake and recompile.
//
// Contract that every backend must expose (same surface as BigIntNTTBatch):
//   ints: n_limbs, padded, n_sum-equivalent (via padded), n_batch
//   Data64* d_buf_A;                       // work slot in the transformed domain
//   ntt_A / ntt_B / ntt_AB(src,len)        // forward transform of variable operand(s)
//   fwd_A()                                // forward of an already-filled d_buf_A
//   pmul() / psq() / pmul_ext(pre)         // pointwise (A*B, A², A*precomputed)
//   intt_A()                               // inverse transform
//   carry_to_limbs(out, out_len)           // domain → normalized limbs (with carry)
//   vadd_raw_buf / carry_after_vadd / add_raw_buf_and_carry  // raw add + carry (REDC)
//   pmul_and_intt / psq_and_intt / pmul_ext_and_intt         // composites
//   schoolbook_mul / schoolbook_sq         // direct convolution (small n)
//
// Backends that don't support some operation must provide the equivalent or
// abort explicitly.
#pragma once

#include "config.h" // MUL_ALG + MUL_* identifiers (via constants.h)

#ifndef MUL_ALG
#error "MUL_ALG not defined (params.cmake → config.h). Use MUL_SCHOOLBOOK | MUL_MERGE_GPUNTT | MUL_4STEP_GPUNTT."
#endif

// Selection of the multiplication class. SCHOOLBOOK and NTT_MERGE share the
// "merge" class (BigIntNTTBatch) — the modular reduction always uses NTT, and the
// merge class provides both the NTT and the schoolbook_*. Only NTT_4STEP swaps the class.
#if MUL_ALG == MUL_4STEP_GPUNTT
#include "ops/mul/ntt_4step.cuh"   // Ntt4StepBatch class (same API, radix algorithm)
using Multiplier = Ntt4StepBatch;
#elif MUL_ALG == MUL_FFT_CUFFT
#include "ops/mul/fft_cufft.cuh"   // FftCuFFTBatch class (complex FFT via cuFFT)
using Multiplier = FftCuFFTBatch;
#elif MUL_ALG == MUL_FFT_GPUFFT
#include "ops/mul/fft_gpufft.cuh"  // FftGpuFftBatch class (GPU-FFT C2C, merge style)
using Multiplier = FftGpuFftBatch;
#elif MUL_ALG == MUL_FFNT_GPUFFT
#include "ops/mul/fft_ffnt.cuh"    // FftFFNTBatch class (GPU-FFT negacyclic real)
using Multiplier = FftFFNTBatch;
#elif MUL_ALG == MUL_MERGE_GPUNTT || MUL_ALG == MUL_SCHOOLBOOK
#include "ops/mul/ntt_merge.cuh"   // BigIntNTTBatch class (GPU-NTT merge backend)
using Multiplier = BigIntNTTBatch;
#else
#error "MUL_ALG invalid. Use MUL_SCHOOLBOOK | MUL_MERGE_GPUNTT | MUL_4STEP_GPUNTT | MUL_FFT_CUFFT."
#endif
