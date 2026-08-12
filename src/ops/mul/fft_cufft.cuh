/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/mul/fft_cufft.cuh
 * ROLE   complex FFT via cuFFT
 *
 * HOW    Double-precision complex transform, padded = 2*fft_len, with a
 *        round-and-scatter pass after the inverse to get integers back.
 *
 * NOTE   APPROXIMATE. Correct only while the largest coefficient fits the
 *        52-bit mantissa; the constructor guards that and refuses the
 *        configuration otherwise. Same public surface as the other backends
 *        (see ops/mul/multiplier.cuh), so the reductions and the
 *        orchestrator never learn which one is active.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cufft.h>

#ifndef LIMB_BITS
#define LIMB_BITS 16
#endif
#ifndef LIMB_MASK
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1ULL)
#endif

// Data64 is the limb type used throughout the project (uint64). Here we do not depend
// on GPU-NTT, so we define it locally if it has not yet come from another header.
#ifndef GPUNTT_DATA64_ALIAS
#define GPUNTT_DATA64_ALIAS
using Data64 = unsigned long long;
#endif

#ifndef NTT_HELPERS_DEFINED
#define NTT_HELPERS_DEFINED
inline int limbs_for_digits(int decimal_digits)
{
    return (int)((decimal_digits * 3.32193 + LIMB_BITS - 1) / LIMB_BITS) + 4;
}
inline int next_pow2_ntt(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
#endif

#include "ops/limb_storage.cuh"

struct FftCuFFTBatch
{
    int n_limbs, padded, logn, n_batch;
    int fft_len;
    int spec_len;

    Data64 *d_buf_AB = nullptr;
    Data64 *d_buf_A = nullptr;
    Data64 *d_buf_B = nullptr;
    double *d_in = nullptr;
    double *d_real = nullptr;
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr;
    int    *d_first_tile = nullptr;
#endif

    cufftHandle plan_r2c_n = 0;
    cufftHandle plan_r2c_2n = 0;
    cufftHandle plan_c2r_n = 0;

    explicit FftCuFFTBatch(int n_limbs_, int n_batch_);
    ~FftCuFFTBatch();

    LimbT *raw_coeffs() { return reinterpret_cast<LimbT *>(d_real); }

    // Forward transforms
    void ntt_A(const LimbT *d_src, int n_src, cudaStream_t s = 0);
    void ntt_B(const LimbT *d_src, int n_src, cudaStream_t s = 0);
    void ntt_AB(const LimbT *d_srcA, const LimbT *d_srcB, int n_src, cudaStream_t s = 0);
    void fwd_A(cudaStream_t s = 0);

    // Pointwise (complex)
    void pmul(cudaStream_t s = 0);
    void psq(cudaStream_t s = 0);
    void pmul_ext(const Data64 *d_ext, cudaStream_t s = 0);

    // Inverse → round → integers in d_buf_A
    void intt_A(cudaStream_t s = 0);

    // Composite
    void pmul_and_intt(cudaStream_t s = 0);
    void psq_and_intt(cudaStream_t s = 0);
    void pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s = 0);

    // Schoolbook (does not use FFT; operates directly on limbs → d_buf_A integers)
    void schoolbook_mul(const LimbT *d_A, const LimbT *d_B, int n_src, cudaStream_t s = 0);
    void schoolbook_sq(const LimbT *d_A, int n_src, cudaStream_t s = 0);

    // Carry / sum (defined in ops/carry/carry_norm.cu, agnostic: Multiplier::)
    void carry_to_limbs(LimbT *d_out, int n_out, cudaStream_t s = 0);
    void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, int n_passes, cudaStream_t s = 0);
    void vadd_raw_buf(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    void carry_after_vadd(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    void add_raw_buf_and_carry(LimbT *d_dst, int n_dst, cudaStream_t s = 0);

    FftCuFFTBatch(const FftCuFFTBatch &) = delete;
    FftCuFFTBatch &operator=(const FftCuFFTBatch &) = delete;
};

void carry_stats_print_and_reset();
