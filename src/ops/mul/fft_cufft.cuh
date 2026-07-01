#pragma once
// ops/mul/fft_cufft.cuh — big-int multiplication backend via complex FFT (cuFFT).
//
// Same public interface as BigIntNTTBatch (see ops/mul/multiplier.cuh). Differs
// by being APPROXIMATE: uses double-complex, so the rounding of the coefficients
// is only correct as long as the largest coefficient fits in the 52-bit mantissa
// (precision guard in the constructor).
//
// Layout (trick to reuse the generic precomputation of BatchModCtx):
//   padded = 2 * fft_len  (Data64 units). d_buf_A is reinterpreted as
//   cufftDoubleComplex[fft_len] per polynomial → the generic memcpy that caches the
//   transform (d_ntt_N/d_ntt_mu) copies exactly the complex bytes.
//   After intt_A, we round to integers and scatter into d_buf_A (stride
//   padded) so the shared carry (ops/carry) works as in the NTT backends.

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

#include "ops/limb_storage.cuh" // LimbT (needs Data64)

struct FftCuFFTBatch
{
    int n_limbs, padded, logn, n_batch;
    int fft_len;  // real-FFT length (real points). padded = 2*fft_len (Data64).
    int spec_len; // = fft_len/2 + 1: # of independent complex outputs (Hermitian R2C).

    // REAL-FFT layout (R2C / C2R):
    //   • d_in [2*n_batch*fft_len] doubles — zero-padded real INPUT (A,B), distance fft_len.
    //   • d_buf_A/B — Hermitian spectra (spec_len complex), stored at distance fft_len
    //     complex (= padded Data64) so the generic d_ntt_N cache (stride padded) matches.
    //   • d_real [n_batch*padded] doubles — real OUTPUT of C2R = raw_coeffs(), stride padded.
    Data64 *d_buf_AB = nullptr; // [2 * n_batch * padded] Data64 (A,B spectra, distance fft_len cplx)
    Data64 *d_buf_A = nullptr;  // = d_buf_AB
    Data64 *d_buf_B = nullptr;  // = d_buf_AB + n_batch*padded
    double *d_in = nullptr;     // [2 * n_batch * fft_len] real input (distance fft_len)
    double *d_real = nullptr;   // [n_batch * padded] real output (raw_coeffs(), stride padded)
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr;
    int    *d_first_tile = nullptr; // [n_batch] first tile with non-zero residual (n_tiles = none)
#endif

    cufftHandle plan_r2c_n = 0;  // D2Z batch = n_batch
    cufftHandle plan_r2c_2n = 0; // D2Z batch = 2*n_batch
    cufftHandle plan_c2r_n = 0;  // Z2D batch = n_batch

    explicit FftCuFFTBatch(int n_limbs_, int n_batch_);
    ~FftCuFFTBatch();

    // Buffer holding the raw (un-normalized, scaled) real coefficients the carry
    // layer reads — the inverse-FFT output, stride = padded. NOT the complex d_buf_A.
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
