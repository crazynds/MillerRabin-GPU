#pragma once
// ops/mul/fft_ffnt.cuh — big-int multiplication backend via FFNT (GPU-FFT).
//
// FFNT = Fast Fourier Negacyclic Transform: REAL FFT for convolution mod X^n+1.
// Real input (limbs) → half the complex size (n/2) → ~2x more efficient than
// the C2C FFT. APPROXIMATE (double): same precision guard as the other FFT backends.
//
// For linear big-int multiplication we use n = fft_len ≥ 2·n_limbs, so the
// negacyclic convolution does NOT wrap around (high part zero) ⇒ result = linear.
//
// Layout (lib contract, from the R_R example):
//   • real buffer d_real: n contiguous reals per polynomial (coef i at position i).
//   • transformed domain = device_temp: n/2 complex per polynomial. This is
//     d_buf_A (reinterpreted), so that padded = n = fft_len (Data64).
//   • forward: GPU_FFNT(real → temp); pmul on temp; inverse: GPU_FFNT(temp → real).
//   After the inverse we round the reals and write integers into d_buf_A (carry).

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

struct FftFFNTBatch
{
    int n_limbs, padded, logn, n_batch;
    int fft_len; // = n = padded (negacyclic). transformed domain = n/2 complex.

    Data64 *d_buf_AB = nullptr; // [2*n_batch*padded] = transformed domain (temp) of A,B
    Data64 *d_buf_A = nullptr;
    Data64 *d_buf_B = nullptr;
    Data64 *d_tile_carry = nullptr;
    double *d_real = nullptr;   // [2*n_batch*fft_len] reals (FFNT I/O)

    Complex64 *d_root_fwd = nullptr;
    Complex64 *d_root_inv = nullptr;
    Complex64 *d_twist = nullptr;
    Complex64 *d_untwist = nullptr;
    double n_inv = 1.0;

    explicit FftFFNTBatch(int n_limbs_, int n_batch_);
    ~FftFFNTBatch();

    void ntt_A(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_B(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s = 0);
    void fwd_A(cudaStream_t s = 0);

    void pmul(cudaStream_t s = 0);
    void psq(cudaStream_t s = 0);
    void pmul_ext(const Data64 *d_ext, cudaStream_t s = 0);

    void intt_A(cudaStream_t s = 0);

    void pmul_and_intt(cudaStream_t s = 0);
    void psq_and_intt(cudaStream_t s = 0);
    void pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s = 0);

    void schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s = 0);
    void schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s = 0);

    void carry_to_limbs(Data64 *d_out, int n_out, cudaStream_t s = 0);
    void add_and_carry(Data64 *d_a, const Data64 *d_b, int n, int n_passes, cudaStream_t s = 0);
    void vadd_raw_buf(Data64 *d_dst, int n_dst, cudaStream_t s = 0);
    void carry_after_vadd(Data64 *d_dst, int n_dst, cudaStream_t s = 0);
    void add_raw_buf_and_carry(Data64 *d_dst, int n_dst, cudaStream_t s = 0);

    FftFFNTBatch(const FftFFNTBatch &) = delete;
    FftFFNTBatch &operator=(const FftFFNTBatch &) = delete;

private:
    // Runs the FFNT forward (real d_real → temp tbuf) or inverse (temp tbuf → d_real)
    // over `batch` polynomials. `rbuf`/`tbuf` point to the start of the blocks.
    void run_ffnt(double *rbuf, Data64 *tbuf, bool fwd, int batch, cudaStream_t s);
};

void carry_stats_print_and_reset();
