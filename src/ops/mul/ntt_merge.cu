/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/mul/ntt_merge.cu
 * ROLE   big-integer multiplication over the GPU-NTT "merge" transform
 *
 * HOW    Builds the twiddle tables once per batch and drives the fused
 *        entry points in ntt/. The gather, the pointwise stage and the
 *        Barrett shift all ride inside the transform kernels rather than
 *        costing separate passes.
 *
 * CHANGELOG
 *   2026-08-11  Builds the Shoup quotient tables (MR_NTT_SHOUP) and routes
 *               the in-place entry points through the local kernels, so no
 *               configuration falls back to the stock butterflies.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "config.h"
#include "ops/mul/ntt_merge.cuh"
#include "ops/mul/ntt_check.cuh"
#include "ntt/ntt_merge_intt_fused.cuh"
#include "ntt/ntt_merge_ntt_fused.cuh"
#include "ntt/shoup_butterfly.cuh"
#include <stdexcept>
#include <string>
#include "util/cuda_check.cuh"


// ── NTT kernels ───────────────────────────────────────────────────────────────

#ifndef MR_NTT_FUSED_PAD
__global__ static void load_padded_batch(Data64 *__restrict__ dst,
                                         const LimbT *__restrict__ src,
                                         int n_src, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;
    dst[cand * padded + j] = (j < n_src) ? (Data64)limb_ld(src[cand * n_src + j]) : 0ULL;
}
#endif

__global__ static void pmul_batch(Data64 *__restrict__ a, const Data64 *__restrict__ b,
                                  int total, Data64 p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    a[i] = (Data64)((__uint128_t)a[i] * b[i] % (__uint128_t)p);
}

__global__ static void psq_batch(Data64 *__restrict__ a, int total, Data64 p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    a[i] = (Data64)((__uint128_t)a[i] * a[i] % (__uint128_t)p);
}

// ── BigIntNTTBatch ────────────────────────────────────────────────────────────

BigIntNTTBatch::BigIntNTTBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_), padded(next_pow2_ntt(2 * n_limbs_)), logn(__builtin_ctz(next_pow2_ntt(2 * n_limbs_))), n_batch(n_batch_)
{
    if (logn < 1 || logn > 28)
        throw std::runtime_error(
            "[ntt_merge] logn=" + std::to_string(logn) +
            " outside [1,28] (GPU-NTT limit for Data64).");

    NTTParameters<Data64> params(logn, ReductionPolynomial::X_N_minus);
    p_val = params.modulus.value;
    n_inv = params.n_inv;
    modulus = params.modulus;

    check_ntt_precision(padded, p_val);
    check_ntt_padding_efficiency(n_limbs, padded, p_val);

    auto fwd_h = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
    auto inv_h = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);

    const size_t tbytes = fwd_h.size() * sizeof(Root64);
    const size_t pbytes = (size_t)n_batch * padded * sizeof(Data64);
    CU(cudaMalloc(&d_fwd_table, tbytes));
    CU(cudaMalloc(&d_inv_table, tbytes));
    CU(cudaMalloc(&d_buf_AB, 2 * pbytes));
    d_buf_A = d_buf_AB;
    d_buf_B = d_buf_AB + (size_t)n_batch * padded;
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    {
        int n_tiles_max = (padded + MR_CARRY_TILE - 1) / MR_CARRY_TILE;
        CU(cudaMalloc(&d_tile_carry, (size_t)n_batch * n_tiles_max * sizeof(Data64)));
        CU(cudaMalloc(&d_first_tile, (size_t)n_batch * sizeof(int)));
    }
#endif

    CU(cudaMemcpy(d_fwd_table, fwd_h.data(), tbytes, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_inv_table, inv_h.data(), tbytes, cudaMemcpyHostToDevice));

#ifdef MR_NTT_SHOUP
    {
        auto fwd_q = shoup_table_build(fwd_h, p_val);
        auto inv_q = shoup_table_build(inv_h, p_val);
        const size_t qbytes = fwd_q.size() * sizeof(Data64);
        CU(cudaMalloc(&d_fwd_shoup, qbytes));
        CU(cudaMalloc(&d_inv_shoup, qbytes));
        CU(cudaMemcpy(d_fwd_shoup, fwd_q.data(), qbytes, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_inv_shoup, inv_q.data(), qbytes, cudaMemcpyHostToDevice));
        n_inv_shoup = shoup_scalar((Data64)n_inv, p_val);
    }
#endif
}

BigIntNTTBatch::~BigIntNTTBatch()
{
    cudaFree(d_fwd_table);
    cudaFree(d_inv_table);
    cudaFree(d_fwd_shoup);
    cudaFree(d_inv_shoup);
    cudaFree(d_buf_AB);
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    cudaFree(d_tile_carry);
    cudaFree(d_first_tile);
#endif
}

ntt_configuration<Data64> BigIntNTTBatch::make_cfg(type t, cudaStream_t s)
{
    return {
        .n_power = logn,
        .ntt_type = t,
        .ntt_layout = (logn >= 10) ? PerPolynomial : GPUNTT_NTT_LAYOUT,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = n_inv,
        .stream = s};
}

void BigIntNTTBatch::ntt_A(const LimbT *d_src, int n_src, cudaStream_t s)
{
#ifdef MR_NTT_FUSED_PAD
    GPU_NTT_ZeroPadLoad(d_buf_A, d_src, n_src, d_fwd_table, d_fwd_shoup, modulus,
                        make_cfg(FORWARD, s), n_batch);
#else
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_src, n_src, padded, n_batch);
    GPU_NTT_Inplace_Shoup(d_buf_A, d_fwd_table, d_fwd_shoup, modulus, make_cfg(FORWARD, s), n_batch);
#endif
}

void BigIntNTTBatch::ntt_A_shifted(const LimbT *d_src, const int *d_bark, int delta,
                                   int n_out, int n_src, cudaStream_t s)
{
#ifdef MR_NTT_FUSED_SHIFT
    GPU_NTT_ShiftPadLoad(d_buf_A, d_src, d_bark, delta, n_out, n_src,
                         d_fwd_table, d_fwd_shoup, modulus, make_cfg(FORWARD, s), n_batch);
#else
    (void)d_src; (void)d_bark; (void)delta; (void)n_out; (void)n_src; (void)s;
    throw std::runtime_error("[ntt_merge] ntt_A_shifted requires MR_NTT_FUSED_SHIFT");
#endif
}

void BigIntNTTBatch::ntt_B(const LimbT *d_src, int n_src, cudaStream_t s)
{
#ifdef MR_NTT_FUSED_PAD
    GPU_NTT_ZeroPadLoad(d_buf_B, d_src, n_src, d_fwd_table, d_fwd_shoup, modulus,
                        make_cfg(FORWARD, s), n_batch);
#else
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_B, d_src, n_src, padded, n_batch);
    GPU_NTT_Inplace_Shoup(d_buf_B, d_fwd_table, d_fwd_shoup, modulus, make_cfg(FORWARD, s), n_batch);
#endif
}

void BigIntNTTBatch::ntt_AB(const LimbT *d_srcA, const LimbT *d_srcB, int n_src, cudaStream_t s)
{
#ifdef MR_NTT_FUSED_PAD
    GPU_NTT_ZeroPadLoad2(d_buf_A, d_srcA, d_srcB, n_src, n_batch,
                         d_fwd_table, d_fwd_shoup, modulus, make_cfg(FORWARD, s), 2 * n_batch);
#else
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_srcA, n_src, padded, n_batch);
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_B, d_srcB, n_src, padded, n_batch);
    GPU_NTT_Inplace_Shoup(d_buf_A, d_fwd_table, d_fwd_shoup, modulus, make_cfg(FORWARD, s), 2 * n_batch);
#endif
}

void BigIntNTTBatch::fwd_A(cudaStream_t s)
{
    GPU_NTT_Inplace_Shoup(d_buf_A, d_fwd_table, d_fwd_shoup, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::pmul(cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_buf_B, total, p_val);
}

void BigIntNTTBatch::psq(cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    psq_batch<<<blk, thr, 0, s>>>(d_buf_A, total, p_val);
}

void BigIntNTTBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_ext, total, p_val);
}

void BigIntNTTBatch::intt_A(cudaStream_t s)
{
    GPU_INTT_Inplace_Shoup(d_buf_A, d_inv_table, d_inv_shoup, n_inv_shoup, modulus, make_cfg(INVERSE, s), n_batch);
}

void BigIntNTTBatch::pmul_and_intt(cudaStream_t s)
{
#ifdef MR_NTT_FUSED_PMUL
    GPU_INTT_Inplace_PreMul(d_buf_A, d_buf_B, d_inv_table, d_inv_shoup, n_inv_shoup, modulus, make_cfg(INVERSE, s), n_batch);
#else
    pmul(s);
    intt_A(s);
#endif
}

void BigIntNTTBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s)
{
#ifdef MR_NTT_FUSED_PMUL
    GPU_INTT_Inplace_PreMul(d_buf_A, d_ext, d_inv_table, d_inv_shoup, n_inv_shoup, modulus, make_cfg(INVERSE, s), n_batch);
#else
    pmul_ext(d_ext, s);
    intt_A(s);
#endif
}

void BigIntNTTBatch::psq_and_intt(cudaStream_t s)
{
#ifdef MR_NTT_FUSED_PMUL
    GPU_INTT_Inplace_PreSq(d_buf_A, d_inv_table, d_inv_shoup, n_inv_shoup, modulus, make_cfg(INVERSE, s), n_batch);
#else
    psq(s);
    intt_A(s);
#endif
}

// ── Schoolbook (MUL_SCHOOLBOOK) ─────────────────────────────────────
//
// Direct O(n²) polynomial convolution: each thread computes one output coefficient.
// Writes to d_buf_A with stride=padded — compatible with carry_to_limbs().
//
// Thread = one output coefficient j ∈ [0, padded).
// Inner loop sums A[i]*B[j-i] for i in [max(0,j-n+1), min(j+1,n)).
// Overflow: A[i],B[j-i] ≤ 2^16-1, loop ≤ n iterations, acc ≤ n*(2^16)² < 2^49 for
// n ≤ 2^17. Fits in uint64. ✓

__global__ static void schoolbook_mul_kernel(
    Data64 *__restrict__ d_buf_A,
    const LimbT *__restrict__ d_A,
    const LimbT *__restrict__ d_B,
    int n_limbs, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;

    if (j >= 2 * n_limbs)
    {
        d_buf_A[(size_t)cand * padded + j] = 0ULL;
        return;
    }

    const LimbT *A = d_A + (size_t)cand * n_limbs;
    const LimbT *B = d_B + (size_t)cand * n_limbs;
    uint64_t acc = 0;
    int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
    int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
    for (int i = i_lo; i < i_hi; i++)
        acc += limb_ld(A[i]) * limb_ld(B[j - i]);
    d_buf_A[(size_t)cand * padded + j] = acc;
}

// Squaring schoolbook with symmetry optimization: pairs (i, j-i) count 2x,
// except the middle term when j is even.
__global__ static void schoolbook_sq_kernel(
    Data64 *__restrict__ d_buf_A,
    const LimbT *__restrict__ d_A,
    int n_limbs, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;

    if (j >= 2 * n_limbs)
    {
        d_buf_A[(size_t)cand * padded + j] = 0ULL;
        return;
    }

    const LimbT *A = d_A + (size_t)cand * n_limbs;
    uint64_t acc = 0;
    int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
    int i_hi_excl = (j < n_limbs) ? j + 1 : n_limbs;
    int i_cross = (j + 1) / 2;
    for (int i = i_lo; i < i_cross && i < i_hi_excl; i++)
        acc += 2ULL * limb_ld(A[i]) * limb_ld(A[j - i]);
    if (j % 2 == 0)
    {
        int m = j / 2;
        if (m >= i_lo && m < i_hi_excl)
            acc += limb_ld(A[m]) * limb_ld(A[m]);
    }
    d_buf_A[(size_t)cand * padded + j] = acc;
}

void BigIntNTTBatch::schoolbook_mul(const LimbT *d_A, const LimbT *d_B, int n_src,
                                    cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_A, d_B, n_src, padded, n_batch);
}

void BigIntNTTBatch::schoolbook_sq(const LimbT *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_A, n_src, padded, n_batch);
}

// ─────────────────────────────────────────────────────────────────────────────
