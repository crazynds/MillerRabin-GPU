#pragma once
// ops/mul/fft_gpufft.cuh — big-int multiplication backend via GPU-FFT (C2C, "merge").
//
// Structure identical to the cuFFT backend (ops/mul/fft_cufft.cuh): approximate (double),
// padded = 2*fft_len, d_buf_A reinterpreted as complex, round+scatter post-INTT.
// Differs by using gpufft::GPU_FFT (in-place) + the lib's root tables; the INVERSE
// already normalizes by 1/N via cfg.mod_inverse.

#include <cstdint>
#include <cuda_runtime.h>
#include "gpufft/fft.cuh"
#include "gpufft/fft_cpu.cuh"

using namespace gpufft;

#ifndef LIMB_BITS
#define LIMB_BITS 16
#endif
#ifndef LIMB_MASK
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1ULL)
#endif
#ifndef GPUNTT_DATA64_ALIAS
#define GPUNTT_DATA64_ALIAS
using Data64 = unsigned long long;
#endif
#ifndef NTT_HELPERS_DEFINED
#define NTT_HELPERS_DEFINED
inline int limbs_for_digits(int decimal_digits)
{ return (int)((decimal_digits * 3.32193 + LIMB_BITS - 1) / LIMB_BITS) + 4; }
inline int next_pow2_ntt(int n) { int p = 1; while (p < n) p <<= 1; return p; }
#endif

#include "ops/limb_storage.cuh" // LimbT (needs Data64)

struct FftGpuFftBatch
{
    int n_limbs, padded, logn, n_batch;
    int fft_len; // = padded / 2 (FFT size, in complex elements)

    Data64 *d_buf_AB = nullptr;   // [2 * n_batch * padded] = 2*n_batch*fft_len complex
    Data64 *d_buf_A = nullptr;
    Data64 *d_buf_B = nullptr;
    double *d_real = nullptr;     // [n_batch * padded] real coefficients (post-INTT, stride padded)
    Data64 *d_cplx_tmp = nullptr; // [n_batch * padded] complex (fwd_A staging)
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr;
    int    *d_first_tile = nullptr; // [n_batch] first tile with non-zero residual (n_tiles = none)
#endif

    Complex64 *d_root_fwd = nullptr; // root table (forward)
    Complex64 *d_root_inv = nullptr; // root table (inverse)
    int root_len = 0;
    double n_inv = 1.0; // 1/N to normalize the INVERSE (cfg.mod_inverse)

    explicit FftGpuFftBatch(int n_limbs_, int n_batch_);
    ~FftGpuFftBatch();

    // Raw real coefficients the carry layer reads (inverse-FFT output, stride padded).
    LimbT *raw_coeffs() { return reinterpret_cast<LimbT *>(d_real); }

    void ntt_A(const LimbT *d_src, int n_src, cudaStream_t s = 0);
    void ntt_B(const LimbT *d_src, int n_src, cudaStream_t s = 0);
    void ntt_AB(const LimbT *d_srcA, const LimbT *d_srcB, int n_src, cudaStream_t s = 0);
    void fwd_A(cudaStream_t s = 0);

    void pmul(cudaStream_t s = 0);
    void psq(cudaStream_t s = 0);
    void pmul_ext(const Data64 *d_ext, cudaStream_t s = 0);

    void intt_A(cudaStream_t s = 0);

    void pmul_and_intt(cudaStream_t s = 0);
    void psq_and_intt(cudaStream_t s = 0);
    void pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s = 0);

    void schoolbook_mul(const LimbT *d_A, const LimbT *d_B, int n_src, cudaStream_t s = 0);
    void schoolbook_sq(const LimbT *d_A, int n_src, cudaStream_t s = 0);

    void carry_to_limbs(LimbT *d_out, int n_out, cudaStream_t s = 0);
    void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, int n_passes, cudaStream_t s = 0);
    void vadd_raw_buf(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    void carry_after_vadd(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    void add_raw_buf_and_carry(LimbT *d_dst, int n_dst, cudaStream_t s = 0);

    FftGpuFftBatch(const FftGpuFftBatch &) = delete;
    FftGpuFftBatch &operator=(const FftGpuFftBatch &) = delete;

private:
    void run_fft(Data64 *buf, bool fwd, int batch, cudaStream_t s);
};

void carry_stats_print_and_reset();
