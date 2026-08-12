/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/mul/fft_ffnt.cuh
 * ROLE   real negacyclic FFT (FFNT) via GPU-FFT
 *
 * HOW    A real input of limbs maps to half the complex size (n/2), which
 *        makes it roughly twice as efficient as the C2C transform for the
 *        same convolution.
 *
 * NOTE   APPROXIMATE, with the same mantissa guard as the other FFT
 *        backends. Same public surface as the other backends (see
 *        ops/mul/multiplier.cuh), so the reductions and the orchestrator
 *        never learn which one is active.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

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

#include "ops/limb_storage.cuh"

struct FftFFNTBatch
{
    int n_limbs, padded, logn, n_batch;
    int fft_len;

    Data64 *d_buf_AB = nullptr;
    Data64 *d_buf_A = nullptr;
    Data64 *d_buf_B = nullptr;
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr;
    int    *d_first_tile = nullptr;
#endif
    double *d_real = nullptr;

    Complex64 *d_root_fwd = nullptr;
    Complex64 *d_root_inv = nullptr;
    Complex64 *d_twist = nullptr;
    Complex64 *d_untwist = nullptr;
    double n_inv = 1.0;

    explicit FftFFNTBatch(int n_limbs_, int n_batch_);
    ~FftFFNTBatch();

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

    FftFFNTBatch(const FftFFNTBatch &) = delete;
    FftFFNTBatch &operator=(const FftFFNTBatch &) = delete;

private:
    // Runs the FFNT forward (real d_real → temp tbuf) or inverse (temp tbuf → d_real)
    // over `batch` polynomials. `rbuf`/`tbuf` point to the start of the blocks.
    void run_ffnt(double *rbuf, Data64 *tbuf, bool fwd, int batch, cudaStream_t s);
};

void carry_stats_print_and_reset();
