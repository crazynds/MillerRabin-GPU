// ops/mul/fft_gpufft.cu — big-int multiplication via GPU-FFT (C2C, double).
#include "config.h"
#include "ops/mul/fft_gpufft.cuh"
#include <stdexcept>
#include <string>
#include <vector>

#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── kernels (double2 for load/round; Complex64 for pointwise via operators) ────

__global__ static void load_complex(double2 *__restrict__ dst, const Data64 *__restrict__ src,
                                    int n_src, int fft_len, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    double v = (j < n_src) ? (double)src[(size_t)cand * n_src + j] : 0.0;
    dst[(size_t)cand * fft_len + j] = make_double2(v, 0.0);
}

__global__ static void load_complex_from_buf(double2 *__restrict__ dst, const Data64 *__restrict__ buf,
                                             int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len)
        return;
    dst[(size_t)cand * fft_len + j] = make_double2((double)buf[(size_t)cand * padded + j], 0.0);
}

__global__ static void cmul_kernel(Complex64 *__restrict__ a, const Complex64 *__restrict__ b, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    a[i] = a[i] * b[i];
}

__global__ static void csq_kernel(Complex64 *__restrict__ a, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    a[i] = a[i] * a[i];
}

// INVERSE already normalized by the lib (mod_inverse = 1/N). Reads real part, rounds.
__global__ static void round_extract(Data64 *__restrict__ d_int, const double2 *__restrict__ src,
                                     int fft_len, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len) return;
    long long v = llround(src[(size_t)cand * fft_len + j].x);
    d_int[(size_t)cand * fft_len + j] = (Data64)(v < 0 ? 0 : v);
}

__global__ static void scatter_int(Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_int,
                                   int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len) return;
    d_buf_A[(size_t)cand * padded + j] = d_int[(size_t)cand * fft_len + j];
}

__global__ static void schoolbook_mul_kernel(Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_A,
                                             const Data64 *__restrict__ d_B, int n_limbs, int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len) return;
    Data64 out = 0;
    if (j < 2 * n_limbs)
    {
        const Data64 *A = d_A + (size_t)cand * n_limbs, *B = d_B + (size_t)cand * n_limbs;
        uint64_t acc = 0;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
        for (int i = i_lo; i < i_hi; i++) acc += A[i] * B[j - i];
        out = acc;
    }
    d_buf_A[(size_t)cand * padded + j] = out;
}

__global__ static void schoolbook_sq_kernel(Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_A,
                                            int n_limbs, int fft_len, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= fft_len) return;
    Data64 out = 0;
    if (j < 2 * n_limbs)
    {
        const Data64 *A = d_A + (size_t)cand * n_limbs;
        uint64_t acc = 0;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi_excl = (j < n_limbs) ? j + 1 : n_limbs;
        int i_cross = (j + 1) / 2;
        for (int i = i_lo; i < i_cross && i < i_hi_excl; i++) acc += 2ULL * A[i] * A[j - i];
        if (j % 2 == 0) { int m = j / 2; if (m >= i_lo && m < i_hi_excl) acc += A[m] * A[m]; }
        out = acc;
    }
    d_buf_A[(size_t)cand * padded + j] = out;
}

// ── ctor / dtor ───────────────────────────────────────────────────────────────

// GPU-FFT (C2C) only supports n_power ∈ [12,24]. For small n we do over-padding
// up to 2^12 (larger transform; product fits, correctness preserved).
static int clamp_fft_len_gpufft(int n_limbs)
{
    int p = next_pow2_ntt(2 * n_limbs);
    if (p < (1 << 12))
        p = (1 << 12);
    return p;
}

FftGpuFftBatch::FftGpuFftBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_),
      padded(2 * clamp_fft_len_gpufft(n_limbs_)),
      logn(__builtin_ctz(clamp_fft_len_gpufft(n_limbs_))),
      n_batch(n_batch_),
      fft_len(clamp_fft_len_gpufft(n_limbs_))
{
    if (logn > 24)
        throw std::runtime_error(
            "[fft_gpufft] n_power=" + std::to_string(logn) +
            " > 24 (GPU-FFT C2C limit). Use MUL_MERGE_GPUNTT for larger sizes.");

    // Precision guard (double, 52-bit mantissa) — see fft_cufft.cu.
    const double max_coeff = (double)n_limbs * (double)LIMB_MASK * (double)LIMB_MASK;
    if (max_coeff * (4.0 * (double)logn) >= 4503599627370496.0 /*2^52*/)
        throw std::runtime_error(
            "[fft_gpufft] insufficient precision: fft_len·(2^LIMB_BITS-1)²·~4logN exceeds 2^52. "
            "Reduce LIMB_BITS/size or use MUL_MERGE_GPUNTT.");

    // GPU-FFT generator: operates with operands of size fft_len/2 and n_power=logn
    // (transform of size fft_len). Root tables and 1/N come from it.
    FFT<Float64> gen(fft_len / 2);
    std::vector<Complex64> fwd = gen.ReverseRootTable();
    std::vector<Complex64> inv = gen.InverseReverseRootTable();
    n_inv = (double)gen.n_inverse;
    root_len = (int)fwd.size();

    CU(cudaMalloc(&d_root_fwd, (size_t)root_len * sizeof(Complex64)));
    CU(cudaMalloc(&d_root_inv, (size_t)inv.size() * sizeof(Complex64)));
    CU(cudaMemcpy(d_root_fwd, fwd.data(), (size_t)root_len * sizeof(Complex64), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_root_inv, inv.data(), inv.size() * sizeof(Complex64), cudaMemcpyHostToDevice));

    const size_t pb = (size_t)n_batch * padded * sizeof(Data64); // = n_batch*fft_len complex
    CU(cudaMalloc(&d_buf_AB, 2 * pb));
    d_buf_A = d_buf_AB;
    d_buf_B = d_buf_AB + (size_t)n_batch * padded;
    CU(cudaMalloc(&d_int, (size_t)n_batch * fft_len * sizeof(Data64)));
    CU(cudaMalloc(&d_cplx_tmp, pb));
    int n_tiles_max = (padded + CARRY_TILE - 1) / CARRY_TILE;
    CU(cudaMalloc(&d_tile_carry, (size_t)n_batch * n_tiles_max * sizeof(Data64)));
}

FftGpuFftBatch::~FftGpuFftBatch()
{
    cudaFree(d_root_fwd);
    cudaFree(d_root_inv);
    cudaFree(d_buf_AB);
    cudaFree(d_int);
    cudaFree(d_cplx_tmp);
    cudaFree(d_tile_carry);
}

// ── FFT (in-place via GPU_FFT, multiplication=false) ──────────────────────────

void FftGpuFftBatch::run_fft(Data64 *buf, bool fwd, int batch, cudaStream_t s)
{
    fft_configuration<Float64> cfg{};
    cfg.n_power = logn;
    cfg.fft_type = fwd ? FORWARD : INVERSE;
    cfg.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg.zero_padding = false;
    cfg.stream = s;
    if (!fwd)
        cfg.mod_inverse = Complex64(n_inv, 0.0);
    GPU_FFT(reinterpret_cast<Complex64 *>(buf), fwd ? d_root_fwd : d_root_inv, cfg, batch, false);
}

void FftGpuFftBatch::ntt_A(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(reinterpret_cast<double2 *>(d_buf_A), d_src, n_src, fft_len, n_batch);
    run_fft(d_buf_A, /*fwd=*/true, n_batch, s);
}

void FftGpuFftBatch::ntt_B(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(reinterpret_cast<double2 *>(d_buf_B), d_src, n_src, fft_len, n_batch);
    run_fft(d_buf_B, /*fwd=*/true, n_batch, s);
}

void FftGpuFftBatch::ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(reinterpret_cast<double2 *>(d_buf_A), d_srcA, n_src, fft_len, n_batch);
    load_complex<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(reinterpret_cast<double2 *>(d_buf_B), d_srcB, n_src, fft_len, n_batch);
    run_fft(d_buf_A, /*fwd=*/true, 2 * n_batch, s); // A and B contiguous
}

void FftGpuFftBatch::fwd_A(cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_complex_from_buf<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(reinterpret_cast<double2 *>(d_cplx_tmp), d_buf_A, fft_len, padded, n_batch);
    run_fft(d_cplx_tmp, /*fwd=*/true, n_batch, s);
    CU(cudaMemcpyAsync(d_buf_A, d_cplx_tmp, (size_t)n_batch * padded * sizeof(Data64), cudaMemcpyDeviceToDevice, s));
}

void FftGpuFftBatch::pmul(cudaStream_t s)
{
    int total = n_batch * fft_len; constexpr int thr = MR_THR_PMUL;
    cmul_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(reinterpret_cast<Complex64 *>(d_buf_A), reinterpret_cast<Complex64 *>(d_buf_B), total);
}

void FftGpuFftBatch::psq(cudaStream_t s)
{
    int total = n_batch * fft_len; constexpr int thr = MR_THR_PMUL;
    csq_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(reinterpret_cast<Complex64 *>(d_buf_A), total);
}

void FftGpuFftBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    int total = n_batch * fft_len; constexpr int thr = MR_THR_PMUL;
    cmul_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(reinterpret_cast<Complex64 *>(d_buf_A), reinterpret_cast<const Complex64 *>(d_ext), total);
}

void FftGpuFftBatch::intt_A(cudaStream_t s)
{
    run_fft(d_buf_A, /*fwd=*/false, n_batch, s);
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    round_extract<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_int, reinterpret_cast<double2 *>(d_buf_A), fft_len, n_batch);
    scatter_int<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_buf_A, d_int, fft_len, padded, n_batch);
}

void FftGpuFftBatch::pmul_and_intt(cudaStream_t s) { pmul(s); intt_A(s); }
void FftGpuFftBatch::psq_and_intt(cudaStream_t s) { psq(s); intt_A(s); }
void FftGpuFftBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s) { pmul_ext(d_ext, s); intt_A(s); }

void FftGpuFftBatch::schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL; unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_buf_A, d_A, d_B, n_src, fft_len, padded, n_batch);
}

void FftGpuFftBatch::schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL; unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_buf_A, d_A, n_src, fft_len, padded, n_batch);
}
