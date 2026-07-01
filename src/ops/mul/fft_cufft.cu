// ops/mul/fft_cufft.cu — big-int multiplication via REAL FFT (cuFFT R2C/C2R, double).
//
// Uses cufftExecD2Z (forward, real→Hermitian complex) and cufftExecZ2D (inverse,
// Hermitian complex→real). A real FFT of fft_len points has only spec_len=fft_len/2+1
// independent complex outputs, so this does ~half the work/memory of the old C2C path.
//
// Convolution: D2Z(a) .* D2Z(b) → Z2D = fft_len · cyclic_conv(a,b). With fft_len ≥
// 2·n_limbs the cyclic conv equals the linear product. cuFFT transforms are
// un-normalized, so the 1/fft_len factor is folded into the pointwise multiply
// (every intt is preceded by a pointwise) — Z2D then writes the final reals straight
// into d_real (= raw_coeffs()), no extra scale pass.
#include "config.h"
#include "ops/mul/fft_cufft.cuh"
#include <cuComplex.h>
#include <stdexcept>
#include <string>

#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

#define CUFFT_CHECK(expr)                                                                             \
    do                                                                                                \
    {                                                                                                 \
        cufftResult _r = (expr);                                                                      \
        if (_r != CUFFT_SUCCESS)                                                                      \
            throw std::runtime_error(std::string("[cuFFT] " #expr " failed: ") + std::to_string(_r)); \
    } while (0)

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── kernels ─────────────────────────────────────────────────────────────────

__device__ static inline cufftDoubleComplex cxmul(cufftDoubleComplex a, cufftDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Loads limbs (double, LimbT) from d_src [stride n_src] → real input d_in
// [distance fft_len], zero-padding up to fft_len.
__global__ static void load_real(double *__restrict__ dst, const LimbT *__restrict__ src,
                                 int n_src, int fft_len, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    dst[(size_t)cand * fft_len + j] = (j < n_src) ? (double)src[(size_t)cand * n_src + j] : 0.0;
}

// Copies already-real coefficients from d_buf_A (stride padded, filled by extract_low
// in double) into the real input buffer (distance fft_len) — for fwd_A.
__global__ static void real2real(double *__restrict__ dst, const double *__restrict__ buf,
                                 int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    dst[(size_t)cand * fft_len + j] = buf[(size_t)cand * padded + j];
}

// Pointwise over the Hermitian spectrum: spec_len complex per candidate, stored at
// distance fft_len complex. `inv` (= 1/fft_len) folds the inverse-FFT normalization.
__global__ static void cmul_kernel(cufftDoubleComplex *__restrict__ a,
                                   const cufftDoubleComplex *__restrict__ b,
                                   int spec_len, int fft_len, double inv, int n_batch)
{
    int cand = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || k >= spec_len)
        return;
    size_t i = (size_t)cand * fft_len + k;
    cufftDoubleComplex v = cxmul(a[i], b[i]);
    a[i] = make_cuDoubleComplex(v.x * inv, v.y * inv);
}

__global__ static void csq_kernel(cufftDoubleComplex *__restrict__ a,
                                  int spec_len, int fft_len, double inv, int n_batch)
{
    int cand = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || k >= spec_len)
        return;
    size_t i = (size_t)cand * fft_len + k;
    cufftDoubleComplex v = cxmul(a[i], a[i]);
    a[i] = make_cuDoubleComplex(v.x * inv, v.y * inv);
}

// schoolbook is never called when MUL_ALG==MUL_FFT_CUFFT, but its signature must match
// the LimbT contract. Writes into raw_coeffs() (d_real, stride padded).
__global__ static void schoolbook_mul_kernel(
    double *__restrict__ out_raw, const LimbT *__restrict__ d_A,
    const LimbT *__restrict__ d_B, int n_limbs, int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    uint64_t acc = 0;
    if (j < 2 * n_limbs)
    {
        const LimbT *A = d_A + (size_t)cand * n_limbs;
        const LimbT *B = d_B + (size_t)cand * n_limbs;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
        for (int i = i_lo; i < i_hi; i++)
            acc += limb_ld(A[i]) * limb_ld(B[j - i]);
    }
    out_raw[(size_t)cand * padded + j] = (double)acc;
}

__global__ static void schoolbook_sq_kernel(
    double *__restrict__ out_raw, const LimbT *__restrict__ d_A,
    int n_limbs, int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    uint64_t acc = 0;
    if (j < 2 * n_limbs)
    {
        const LimbT *A = d_A + (size_t)cand * n_limbs;
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
    }
    out_raw[(size_t)cand * padded + j] = (double)acc;
}

// ── ctor / dtor ───────────────────────────────────────────────────────────────

FftCuFFTBatch::FftCuFFTBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_),
      padded(2 * next_pow2_ntt(2 * n_limbs_)),
      logn(__builtin_ctz(next_pow2_ntt(2 * n_limbs_))),
      n_batch(n_batch_),
      fft_len(next_pow2_ntt(2 * n_limbs_)),
      spec_len(next_pow2_ntt(2 * n_limbs_) / 2 + 1)
{
    // Precision guard: each convolution coefficient = Σ A[i]·B[k-i], with at
    // most fft_len terms, each ≤ (2^LIMB_BITS−1)². The double FFT accumulates error
    // ~ O(log N)·u (u = 2^-53). To round correctly, max_coeff·(margin) < 2^52.
    const double max_coeff = (double)n_limbs * (double)LIMB_MASK * (double)LIMB_MASK;
    const double err_margin = 4.0 * (double)logn; // conservative growth of the FFT error
    const double mantissa = 4503599627370496.0;   // 2^52
    if (max_coeff * err_margin >= mantissa)
        throw std::runtime_error(
            "[fft_cufft] insufficient precision: fft_len(" + std::to_string(fft_len) +
            ")·(2^" + std::to_string((int)LIMB_BITS) + "-1)²·~4logN exceeds the 52-bit "
                                                       "mantissa of double. Reduce LIMB_BITS/size or use MUL_MERGE_GPUNTT.");

    const size_t pb = (size_t)n_batch * padded * sizeof(Data64); // = n_batch*fft_len complex
    CU(cudaMalloc(&d_buf_AB, 2 * pb));
    d_buf_A = d_buf_AB;
    d_buf_B = d_buf_AB + (size_t)n_batch * padded;
    CU(cudaMalloc(&d_in, (size_t)2 * n_batch * fft_len * sizeof(double)));
    CU(cudaMalloc(&d_real, (size_t)n_batch * padded * sizeof(double)));

#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    int n_tiles_max = (padded + CARRY_TILE - 1) / CARRY_TILE;
    CU(cudaMalloc(&d_tile_carry, (size_t)n_batch * n_tiles_max * sizeof(Data64)));
    CU(cudaMalloc(&d_first_tile, (size_t)n_batch * sizeof(int)));
#endif

    // R2C / C2R plans via cufftPlanMany to control the per-candidate distance:
    //   D2Z: real in distance fft_len → complex out distance fft_len (≥ spec_len),
    //        so spectra sit at distance fft_len complex = padded Data64 (matches d_ntt_N).
    //   Z2D: complex in distance fft_len → real out distance padded (= carry stride).
    int n[1] = {fft_len};
    int re[1] = {fft_len};   // real embed
    int ce[1] = {fft_len};   // complex embed
    int rce[1] = {padded};   // real output embed (Z2D)
    CUFFT_CHECK(cufftPlanMany(&plan_r2c_n, 1, n, re, 1, fft_len, ce, 1, fft_len, CUFFT_D2Z, n_batch));
    CUFFT_CHECK(cufftPlanMany(&plan_r2c_2n, 1, n, re, 1, fft_len, ce, 1, fft_len, CUFFT_D2Z, 2 * n_batch));
    CUFFT_CHECK(cufftPlanMany(&plan_c2r_n, 1, n, ce, 1, fft_len, rce, 1, padded, CUFFT_Z2D, n_batch));
}

FftCuFFTBatch::~FftCuFFTBatch()
{
    if (plan_r2c_n)
        cufftDestroy(plan_r2c_n);
    if (plan_r2c_2n)
        cufftDestroy(plan_r2c_2n);
    if (plan_c2r_n)
        cufftDestroy(plan_c2r_n);
    cudaFree(d_buf_AB);
    cudaFree(d_in);
    cudaFree(d_real);
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    cudaFree(d_tile_carry);
    cudaFree(d_first_tile);
#endif
}

// ── launch helpers ─────────────────────────────────────────────────────
static inline cufftDoubleComplex *cplx(Data64 *p) { return reinterpret_cast<cufftDoubleComplex *>(p); }

// ── forward (R2C) ─────────────────────────────────────────────────────────────

void FftCuFFTBatch::ntt_A(const LimbT *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_in, d_src, n_src, fft_len, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_r2c_n, s));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_n, d_in, cplx(d_buf_A)));
}

void FftCuFFTBatch::ntt_B(const LimbT *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    double *inB = d_in + (size_t)n_batch * fft_len;
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(inB, d_src, n_src, fft_len, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_r2c_n, s));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_n, inB, cplx(d_buf_B)));
}

void FftCuFFTBatch::ntt_AB(const LimbT *d_srcA, const LimbT *d_srcB, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    double *inB = d_in + (size_t)n_batch * fft_len;
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_in, d_srcA, n_src, fft_len, n_batch);
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(inB, d_srcB, n_src, fft_len, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_r2c_2n, s));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_2n, d_in, cplx(d_buf_A))); // A,B spectra contiguous
}

void FftCuFFTBatch::fwd_A(cudaStream_t s)
{
    // d_buf_A holds real coefficients (stride padded, from extract_low). Copy into the
    // real input buffer (distance fft_len), then R2C out-of-place back into d_buf_A.
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    real2real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_in, reinterpret_cast<const double *>(d_buf_A), fft_len, padded, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_r2c_n, s));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_n, d_in, cplx(d_buf_A)));
}

// ── pointwise (Hermitian spectrum, folds 1/fft_len) ───────────────────────────

void FftCuFFTBatch::pmul(cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    dim3 g((unsigned)(spec_len + thr - 1) / thr, (unsigned)n_batch);
    cmul_kernel<<<g, thr, 0, s>>>(cplx(d_buf_A), cplx(d_buf_B), spec_len, fft_len, 1.0 / (double)fft_len, n_batch);
}

void FftCuFFTBatch::psq(cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    dim3 g((unsigned)(spec_len + thr - 1) / thr, (unsigned)n_batch);
    csq_kernel<<<g, thr, 0, s>>>(cplx(d_buf_A), spec_len, fft_len, 1.0 / (double)fft_len, n_batch);
}

void FftCuFFTBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    dim3 g((unsigned)(spec_len + thr - 1) / thr, (unsigned)n_batch);
    cmul_kernel<<<g, thr, 0, s>>>(cplx(d_buf_A), reinterpret_cast<const cufftDoubleComplex *>(d_ext),
                                  spec_len, fft_len, 1.0 / (double)fft_len, n_batch);
}

// ── inverse (C2R) ─────────────────────────────────────────────────────────────

void FftCuFFTBatch::intt_A(cudaStream_t s)
{
    // Z2D writes fft_len reals per candidate at distance padded into d_real (= raw_coeffs()).
    // Normalization already folded into the preceding pointwise; carry rounds via limb_ld.
    CUFFT_CHECK(cufftSetStream(plan_c2r_n, s));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_n, cplx(d_buf_A), d_real));
}

void FftCuFFTBatch::pmul_and_intt(cudaStream_t s)
{
    pmul(s);
    intt_A(s);
}
void FftCuFFTBatch::psq_and_intt(cudaStream_t s)
{
    psq(s);
    intt_A(s);
}
void FftCuFFTBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s)
{
    pmul_ext(d_ext, s);
    intt_A(s);
}

// ── schoolbook (does not use FFT; never called when MUL_ALG==MUL_FFT_CUFFT) ────

void FftCuFFTBatch::schoolbook_mul(const LimbT *d_A, const LimbT *d_B, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_real, d_A, d_B, n_src, fft_len, padded, n_batch);
}

void FftCuFFTBatch::schoolbook_sq(const LimbT *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_real, d_A, n_src, fft_len, padded, n_batch);
}
