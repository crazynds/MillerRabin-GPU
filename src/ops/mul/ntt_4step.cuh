/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/mul/ntt_4step.cuh
 * ROLE   big-integer multiplication over the GPU-NTT "4-step" transform
 *
 * HOW    Radix decomposition with transposes, selected by MUL_ALG ==
 *        MUL_4STEP_GPUNTT. Exact, like the merge backend, but requires
 *        log2(n) in [12, 24] and over-pads to reach it.
 *
 * NOTE   Same public surface as the other backends (see
 *        ops/mul/multiplier.cuh), so the reductions and the orchestrator
 *        never learn which one is active.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include "ops/mul/ntt_4step.cuh"

using namespace gpuntt;

#ifndef LIMB_BITS
#define LIMB_BITS 16
#endif
#ifndef LIMB_MASK
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1ULL)
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

struct Ntt4StepBatch
{
    int n_limbs, padded, logn, n_batch;
    int n1, n2;

    Data64 p_val;
    Ninverse64 n_inv;
    Modulus<Data64> modulus;

    Root64 *d_n1_fwd = nullptr;
    Root64 *d_n2_fwd = nullptr;
    Root64 *d_W_fwd = nullptr;
    Root64 *d_n1_inv = nullptr;
    Root64 *d_n2_inv = nullptr;
    Root64 *d_W_inv = nullptr;

    Modulus<Data64> *d_modulus = nullptr;
    Ninverse64 *d_ninverse = nullptr;

    Data64 *d_buf_AB = nullptr;
    Data64 *d_buf_A = nullptr;
    Data64 *d_buf_B = nullptr;
    Data64 *d_scratch = nullptr;
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    Data64 *d_tile_carry = nullptr;
    int    *d_first_tile = nullptr;
#endif

    explicit Ntt4StepBatch(int n_limbs_, int n_batch_);
    ~Ntt4StepBatch();

    LimbT *raw_coeffs() { return reinterpret_cast<LimbT *>(d_buf_A); }

    // ── Forward transforms ────────────────────────────────────────────────────
    void ntt_A(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_B(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s = 0);
    void fwd_A(cudaStream_t s = 0);

    // ── Pointwise ───────────────────────────────────────────────────────────────
    void pmul(cudaStream_t s = 0);
    void psq(cudaStream_t s = 0);
    void pmul_ext(const Data64 *d_ext, cudaStream_t s = 0);

    // ── Inverse ───────────────────────────────────────────────────────────────
    void intt_A(cudaStream_t s = 0);

    // ── Composites ──────────────────────────────────────────────────────────────
    void pmul_and_intt(cudaStream_t s = 0);
    void psq_and_intt(cudaStream_t s = 0);
    void pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s = 0);

    // ── Schoolbook (MUL_SCHOOLBOOK) ────────────────────────────────────
    void schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s = 0);
    void schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s = 0);

    // ── Carry / add (defined in ops/carry/carry_norm.cu, backend-agnostic) ──────
    void carry_to_limbs(LimbT *d_out, int n_out, cudaStream_t s = 0);
    void add_and_carry(LimbT *d_a, const LimbT *d_b, int n, int n_passes, cudaStream_t s = 0);
    void vadd_raw_buf(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    void carry_after_vadd(LimbT *d_dst, int n_dst, cudaStream_t s = 0);
    void add_raw_buf_and_carry(LimbT *d_dst, int n_dst, cudaStream_t s = 0);

    Ntt4StepBatch(const Ntt4StepBatch &) = delete;
    Ntt4StepBatch &operator=(const Ntt4StepBatch &) = delete;

private:
    // Full 4-step transform (T · NTT · T, 3 ops), forward or inverse.
    // `src` holds the input; the result is written to `dst` (src is used as the
    // ping-pong scratch). src and dst must be distinct buffers. `fwd` chooses the
    // tables/type (FORWARD vs INVERSE).
    void transform(Data64 *src, Data64 *dst, bool fwd, int batch, cudaStream_t s);
};

void carry_stats_print_and_reset();
