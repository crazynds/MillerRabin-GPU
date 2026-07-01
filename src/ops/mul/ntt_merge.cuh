#pragma once
// bigint_ntt.cuh — NTT bigint multiply, 16-bit limbs, single prime, n_batch polys.
//
// Data layout: buf[batch_i * padded + coeff_j]
// A single GPU_NTT_Inplace call processes all n_batch polynomials at once.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"

using namespace gpuntt;

// ── Limb size ───────────────────────────────────────────────────────────────────
// #define LIMB_BITS 32
#ifndef LIMB_BITS
#define LIMB_BITS 16
#endif
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1ULL)

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

static constexpr int CARRY_PASSES_MUL = 4;

#include "ops/limb_storage.cuh" // LimbT, LIMB_IS_REAL, limb_ld/limb_st (needs Data64)

struct BigIntNTTBatch
{
    int n_limbs, padded, logn, n_batch;

    Data64 p_val;
    Ninverse64 n_inv;
    Modulus<Data64> modulus;

    Root64 *d_fwd_table = nullptr;
    Root64 *d_inv_table = nullptr;

    // d_buf_A and d_buf_B are contiguous: d_buf_AB[0..n_batch*padded-1] = A,
    // d_buf_AB[n_batch*padded..2*n_batch*padded-1] = B.
    // This allows calling GPU_NTT_Inplace(d_buf_A, 2*n_batch) to transform
    // A and B in a single kernel launch.
    Data64 *d_buf_AB = nullptr;     // single allocation [2 * n_batch * padded]
    Data64 *d_buf_A = nullptr;      // points into d_buf_AB
    Data64 *d_buf_B = nullptr;      // points into d_buf_AB + n_batch * padded
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr; // [n_batch * n_tiles] inter-tile carry
    int    *d_first_tile = nullptr; // [n_batch] first tile with non-zero residual (n_tiles = none)
#endif

    explicit BigIntNTTBatch(int n_limbs_, int n_batch_);
    ~BigIntNTTBatch();

    // Loads d_src [n_batch * n_src] into d_buf_A with zero-pad up to padded, then NTT
    // Buffer holding the raw INTT coefficients the carry layer reads (= d_buf_A here).
    LimbT *raw_coeffs() { return reinterpret_cast<LimbT *>(d_buf_A); }

    void ntt_A(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    // Same for d_buf_B
    void ntt_B(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    // Loads d_srcA into buf_A and d_srcB into buf_B, then a batched NTT (2*n_batch at once)
    void ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s = 0);

    // d_buf_A = d_buf_A * d_buf_B (pointwise)
    void pmul(cudaStream_t s = 0);
    // d_buf_A = d_buf_A^2 (pointwise)
    void psq(cudaStream_t s = 0);
    // d_buf_A = d_buf_A * d_ext (external, already in NTT domain [n_batch*padded])
    void pmul_ext(const Data64 *d_ext, cudaStream_t s = 0);
    // INTT -> d_buf_A
    void intt_A(cudaStream_t s = 0);
    // Forward NTT only on d_buf_A (already filled externally with zero-pad)
    void fwd_A(cudaStream_t s = 0);

    // Composites kept for compatibility
    void pmul_and_intt(cudaStream_t s = 0);
    void psq_and_intt(cudaStream_t s = 0);
    void pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s = 0);

    // Direct O(n²) polynomial convolution — writes to d_buf_A (stride=padded).
    // Alternative to the ntt_AB + pmul_and_intt pair. Only practical for small n_limbs.
    void schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s = 0);
    // O(n²) squaring version — alternative to ntt_A + psq_and_intt.
    void schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s = 0);

    // Copies d_buf_A -> d_out [n_batch * n_out] and normalizes carries
    void carry_to_limbs(LimbT *d_out, int n_out,
                        cudaStream_t s = 0);
    // d_a += d_b (both [n_batch * n]), then normalizes carries
    void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, int n_passes,
                       cudaStream_t s = 0);
    // d_dst += d_buf_A (raw, stride=padded), without normalizing carries.
    // Not available in CARRY_ALG_SEQUENTIAL (fused with the carry).
    void vadd_raw_buf(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    // Normalizes carries in d_dst after vadd_raw_buf.
    // In CARRY_ALG_SEQUENTIAL it is a no-op (the carry was already done in add_raw_buf_and_carry).
    void carry_after_vadd(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    // Composite version: vadd + carry in one call.
    void add_raw_buf_and_carry(LimbT *d_dst, int n_dst,
                               cudaStream_t s = 0);

    BigIntNTTBatch(const BigIntNTTBatch &) = delete;
    BigIntNTTBatch &operator=(const BigIntNTTBatch &) = delete;

private:
    ntt_configuration<Data64> make_cfg(type t, cudaStream_t s);
};

// Prints and resets the carry iteration counters (MR_ADVANCED_MONITOR).
// No-op when MR_ADVANCED_MONITOR is not defined.
void carry_stats_print_and_reset();
