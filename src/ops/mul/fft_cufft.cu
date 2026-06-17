// ops/mul/fft_cufft.cu — big-int multiplication via complex FFT (cuFFT, double).
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

#define CUFFT_CHECK(expr)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cufftResult _r = (expr);                                                                   \
        if (_r != CUFFT_SUCCESS)                                                                    \
            throw std::runtime_error(std::string("[cuFFT] " #expr " failed: ") + std::to_string(_r)); \
    } while (0)

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── kernels ─────────────────────────────────────────────────────────────────

__device__ static inline cufftDoubleComplex cxmul(cufftDoubleComplex a, cufftDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Loads limbs (Data64) from d_src [stride n_src] → complex (real=limb, imag=0),
// zero-pad up to fft_len.
__global__ static void load_complex(cufftDoubleComplex *__restrict__ dst,
                                    const Data64 *__restrict__ src,
                                    int n_src, int fft_len, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    double v = (j < n_src) ? (double)src[(size_t)cand * n_src + j] : 0.0;
    dst[(size_t)cand * fft_len + j] = make_cuDoubleComplex(v, 0.0);
}

// Same, but reads integers from a buffer with stride `padded` (for fwd_A).
__global__ static void load_complex_from_buf(cufftDoubleComplex *__restrict__ dst,
                                             const Data64 *__restrict__ buf,
                                             int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    double v = (double)buf[(size_t)cand * padded + j];
    dst[(size_t)cand * fft_len + j] = make_cuDoubleComplex(v, 0.0);
}

__global__ static void cmul_kernel(cufftDoubleComplex *__restrict__ a,
                                   const cufftDoubleComplex *__restrict__ b, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    a[i] = cxmul(a[i], b[i]);
}

__global__ static void csq_kernel(cufftDoubleComplex *__restrict__ a, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    a[i] = cxmul(a[i], a[i]);
}

// INTT already done (cufft inverse, does not normalize). Reads real part, scales by 1/fft_len,
// rounds to integer → d_int [stride fft_len].
__global__ static void round_extract(Data64 *__restrict__ d_int,
                                     const cufftDoubleComplex *__restrict__ src,
                                     int fft_len, double scale, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    double r = src[(size_t)cand * fft_len + j].x * scale;
    long long v = llround(r);
    d_int[(size_t)cand * fft_len + j] = (Data64)(v < 0 ? 0 : v);
}

// Scatters integers from d_int [stride fft_len] → d_buf_A [stride padded].
__global__ static void scatter_int(Data64 *__restrict__ d_buf_A,
                                   const Data64 *__restrict__ d_int,
                                   int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    d_buf_A[(size_t)cand * padded + j] = d_int[(size_t)cand * fft_len + j];
}

__global__ static void schoolbook_mul_kernel(
    Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_A,
    const Data64 *__restrict__ d_B, int n_limbs, int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    Data64 out = 0;
    if (j < 2 * n_limbs)
    {
        const Data64 *A = d_A + (size_t)cand * n_limbs;
        const Data64 *B = d_B + (size_t)cand * n_limbs;
        uint64_t acc = 0;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
        for (int i = i_lo; i < i_hi; i++)
            acc += A[i] * B[j - i];
        out = acc;
    }
    d_buf_A[(size_t)cand * padded + j] = out;
}

__global__ static void schoolbook_sq_kernel(
    Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_A,
    int n_limbs, int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    Data64 out = 0;
    if (j < 2 * n_limbs)
    {
        const Data64 *A = d_A + (size_t)cand * n_limbs;
        uint64_t acc = 0;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi_excl = (j < n_limbs) ? j + 1 : n_limbs;
        int i_cross = (j + 1) / 2;
        for (int i = i_lo; i < i_cross && i < i_hi_excl; i++)
            acc += 2ULL * A[i] * A[j - i];
        if (j % 2 == 0)
        {
            int m = j / 2;
            if (m >= i_lo && m < i_hi_excl)
                acc += A[m] * A[m];
        }
        out = acc;
    }
    d_buf_A[(size_t)cand * padded + j] = out;
}

// ── ctor / dtor ───────────────────────────────────────────────────────────────

FftCuFFTBatch::FftCuFFTBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_),
      padded(2 * next_pow2_ntt(2 * n_limbs_)),
      logn(__builtin_ctz(next_pow2_ntt(2 * n_limbs_))),
      n_batch(n_batch_),
      fft_len(next_pow2_ntt(2 * n_limbs_))
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
    CU(cudaMalloc(&d_int, (size_t)n_batch * fft_len * sizeof(Data64)));
    CU(cudaMalloc(&d_cplx_tmp, pb));

    int n_tiles_max = (padded + CARRY_TILE - 1) / CARRY_TILE;
    CU(cudaMalloc(&d_tile_carry, (size_t)n_batch * n_tiles_max * sizeof(Data64)));

    CUFFT_CHECK(cufftPlan1d(&plan_n, fft_len, CUFFT_Z2Z, n_batch));
    CUFFT_CHECK(cufftPlan1d(&plan_2n, fft_len, CUFFT_Z2Z, 2 * n_batch));
}

FftCuFFTBatch::~FftCuFFTBatch()
{
    if (plan_n)
        cufftDestroy(plan_n);
    if (plan_2n)
        cufftDestroy(plan_2n);
    cudaFree(d_buf_AB);
    cudaFree(d_int);
    cudaFree(d_cplx_tmp);
    cudaFree(d_tile_carry);
}

// ── launch helpers ─────────────────────────────────────────────────────

static inline cufftDoubleComplex *cplx(Data64 *p) { return reinterpret_cast<cufftDoubleComplex *>(p); }

// ── forward ─────────────────────────────────────────────────────────────────

void FftCuFFTBatch::ntt_A(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(cplx(d_buf_A), d_src, n_src, fft_len, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_n, s));
    CUFFT_CHECK(cufftExecZ2Z(plan_n, cplx(d_buf_A), cplx(d_buf_A), CUFFT_FORWARD));
}

void FftCuFFTBatch::ntt_B(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(cplx(d_buf_B), d_src, n_src, fft_len, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_n, s));
    CUFFT_CHECK(cufftExecZ2Z(plan_n, cplx(d_buf_B), cplx(d_buf_B), CUFFT_FORWARD));
}

void FftCuFFTBatch::ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(cplx(d_buf_A), d_srcA, n_src, fft_len, n_batch);
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(cplx(d_buf_B), d_srcB, n_src, fft_len, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_2n, s));
    CUFFT_CHECK(cufftExecZ2Z(plan_2n, cplx(d_buf_A), cplx(d_buf_A), CUFFT_FORWARD));
}

void FftCuFFTBatch::fwd_A(cudaStream_t s)
{
    // d_buf_A contains integers (zero-pad) written externally (stride padded).
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex_from_buf<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(cplx(d_cplx_tmp), d_buf_A, fft_len, padded, n_batch);
    CUFFT_CHECK(cufftSetStream(plan_n, s));
    CUFFT_CHECK(cufftExecZ2Z(plan_n, cplx(d_cplx_tmp), cplx(d_cplx_tmp), CUFFT_FORWARD));
    CU(cudaMemcpyAsync(d_buf_A, d_cplx_tmp, (size_t)n_batch * padded * sizeof(Data64),
                       cudaMemcpyDeviceToDevice, s));
}

// ── pointwise ─────────────────────────────────────────────────────────────────

void FftCuFFTBatch::pmul(cudaStream_t s)
{
    int total = n_batch * fft_len;
    constexpr int thr = MR_THR_PMUL;
    cmul_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(cplx(d_buf_A), cplx(d_buf_B), total);
}

void FftCuFFTBatch::psq(cudaStream_t s)
{
    int total = n_batch * fft_len;
    constexpr int thr = MR_THR_PMUL;
    csq_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(cplx(d_buf_A), total);
}

void FftCuFFTBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    int total = n_batch * fft_len;
    constexpr int thr = MR_THR_PMUL;
    cmul_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(
        cplx(d_buf_A), reinterpret_cast<const cufftDoubleComplex *>(d_ext), total);
}

// ── inverse ───────────────────────────────────────────────────────────────────

void FftCuFFTBatch::intt_A(cudaStream_t s)
{
    CUFFT_CHECK(cufftSetStream(plan_n, s));
    CUFFT_CHECK(cufftExecZ2Z(plan_n, cplx(d_buf_A), cplx(d_buf_A), CUFFT_INVERSE));
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    round_extract<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_int, cplx(d_buf_A), fft_len, 1.0 / (double)fft_len, n_batch);
    scatter_int<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_buf_A, d_int, fft_len, padded, n_batch);
}

void FftCuFFTBatch::pmul_and_intt(cudaStream_t s) { pmul(s); intt_A(s); }
void FftCuFFTBatch::psq_and_intt(cudaStream_t s) { psq(s); intt_A(s); }
void FftCuFFTBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s) { pmul_ext(d_ext, s); intt_A(s); }

// ── schoolbook (does not use FFT; never called when MUL_ALG==MUL_FFT_CUFFT, but the
//    interface requires the definition) ───────────────────────────────────────────────

void FftCuFFTBatch::schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_buf_A, d_A, d_B, n_src, fft_len, padded, n_batch);
}

void FftCuFFTBatch::schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_buf_A, d_A, n_src, fft_len, padded, n_batch);
}
