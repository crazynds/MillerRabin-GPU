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

struct FftCuFFTBatch
{
    int n_limbs, padded, logn, n_batch;
    int fft_len; // = padded / 2 (FFT length, in complex elements)

    // Complex buffers (A,B contiguous), reinterpreted from Data64.
    Data64 *d_buf_AB = nullptr;   // [2 * n_batch * padded] Data64 = 2*n_batch*fft_len complex
    Data64 *d_buf_A = nullptr;    // = d_buf_AB
    Data64 *d_buf_B = nullptr;    // = d_buf_AB + n_batch*padded
    Data64 *d_int = nullptr;      // [n_batch * fft_len] integers (post-INTT staging)
    Data64 *d_cplx_tmp = nullptr; // [n_batch * padded] complex (fwd_A staging)
#ifdef CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr;
#endif

    cufftHandle plan_n = 0;  // C2C batch = n_batch
    cufftHandle plan_2n = 0; // C2C batch = 2*n_batch

    explicit FftCuFFTBatch(int n_limbs_, int n_batch_);
    ~FftCuFFTBatch();

    // Forward transforms
    void ntt_A(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_B(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s = 0);
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
    void schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s = 0);
    void schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s = 0);

    // Carry / sum (defined in ops/carry/carry_norm.cu, agnostic: Multiplier::)
    void carry_to_limbs(Data64 *d_out, int n_out, cudaStream_t s = 0);
    void add_and_carry(Data64 *d_a, const Data64 *d_b, int n, int n_passes, cudaStream_t s = 0);
    void vadd_raw_buf(Data64 *d_dst, int n_dst, cudaStream_t s = 0);
    void carry_after_vadd(Data64 *d_dst, int n_dst, cudaStream_t s = 0);
    void add_raw_buf_and_carry(Data64 *d_dst, int n_dst, cudaStream_t s = 0);

    FftCuFFTBatch(const FftCuFFTBatch &) = delete;
    FftCuFFTBatch &operator=(const FftCuFFTBatch &) = delete;
};

void carry_stats_print_and_reset();
