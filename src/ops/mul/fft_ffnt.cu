// ops/mul/fft_ffnt.cu — big-int multiplication via FFNT (GPU-FFT, real negacyclic).
#include "config.h"
#include "ops/mul/fft_ffnt.cuh"
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

// ── kernels ─────────────────────────────────────────────────────────────────

// Real input now: limbs are stored as double (LimbT). Load with zero-padding into
// the contiguous FFNT real buffer — a plain double→double copy (no int conversion).
__global__ static void load_real(double *__restrict__ dst, const LimbT *__restrict__ src,
                                 int n_src, int n, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n) return;
    dst[(size_t)cand * n + j] = (j < n_src) ? (double)src[(size_t)cand * n_src + j] : 0.0;
}

// Copy an already-real-valued d_buf_A (stride=padded, filled by extract_low in double)
// into the contiguous FFNT real buffer (stride=n). Pure double→double copy.
__global__ static void real2real(double *__restrict__ dst, const double *__restrict__ buf,
                                 int n, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n) return;
    dst[(size_t)cand * n + j] = buf[(size_t)cand * padded + j];
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

// NOTE: schoolbook is only used by MUL_SCHOOLBOOK (never by the FFNT path), but its
// signature must match the LimbT contract. It writes into the raw-coefficient buffer
// (raw_coeffs() == d_real) so a subsequent carry_to_limbs would see the result.
__global__ static void schoolbook_mul_kernel(double *__restrict__ out_raw, const LimbT *__restrict__ d_A,
                                             const LimbT *__restrict__ d_B, int n_limbs, int n, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n) return;
    uint64_t acc = 0;
    if (j < 2 * n_limbs)
    {
        const LimbT *A = d_A + (size_t)cand * n_limbs, *B = d_B + (size_t)cand * n_limbs;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
        for (int i = i_lo; i < i_hi; i++) acc += limb_ld(A[i]) * limb_ld(B[j - i]);
    }
    out_raw[(size_t)cand * padded + j] = (double)acc;
}

__global__ static void schoolbook_sq_kernel(double *__restrict__ out_raw, const LimbT *__restrict__ d_A,
                                            int n_limbs, int n, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n) return;
    uint64_t acc = 0;
    if (j < 2 * n_limbs)
    {
        const LimbT *A = d_A + (size_t)cand * n_limbs;
        int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
        int i_hi_excl = (j < n_limbs) ? j + 1 : n_limbs;
        int i_cross = (j + 1) / 2;
        for (int i = i_lo; i < i_cross && i < i_hi_excl; i++) acc += 2ULL * limb_ld(A[i]) * limb_ld(A[j - i]);
        if (j % 2 == 0) { int m = j / 2; if (m >= i_lo && m < i_hi_excl) acc += limb_ld(A[m]) * limb_ld(A[m]); }
    }
    out_raw[(size_t)cand * padded + j] = (double)acc;
}

// ── ctor / dtor ───────────────────────────────────────────────────────────────

// FFNT requires n_power ∈ [12,24] (same as C2C). Over-padding up to 2^12 for small n.
static int clamp_n_ffnt(int n_limbs)
{
    int p = next_pow2_ntt(2 * n_limbs);
    if (p < (1 << 12)) p = (1 << 12);
    return p;
}

FftFFNTBatch::FftFFNTBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_),
      padded(clamp_n_ffnt(n_limbs_)),
      logn(__builtin_ctz(clamp_n_ffnt(n_limbs_))),
      n_batch(n_batch_),
      fft_len(clamp_n_ffnt(n_limbs_))
{
    if (logn > 24)
        throw std::runtime_error(
            "[fft_ffnt] n_power=" + std::to_string(logn) + " > 24 (GPU-FFT limit). Use MUL_MERGE_GPUNTT.");

    const double max_coeff = (double)n_limbs * (double)LIMB_MASK * (double)LIMB_MASK;
    if (max_coeff * (4.0 * (double)logn) >= 4503599627370496.0 /*2^52*/)
        throw std::runtime_error(
            "[fft_ffnt] insufficient precision (52-bit mantissa). Reduce LIMB_BITS/size or use MUL_MERGE_GPUNTT.");

    FFNT<Float64> gen(fft_len);
    std::vector<Complex64> rf = gen.ReverseRootTable_ffnt();
    std::vector<Complex64> ri = gen.InverseReverseRootTable_ffnt();
    std::vector<Complex64> tw = gen.twist_table_ffnt();
    std::vector<Complex64> ut = gen.untwist_table_ffnt();
    // INVERSE normalization = 1/(n/2) (the FFNT runs a complex FFT of n/2 points).
    // Do NOT use gen.n_inverse (= 1/(2n) → result would come out 4× smaller). Value confirmed
    // by the lib's R_R example: n_inverse_new = 1.0/(n>>1).
    n_inv = 1.0 / (double)(fft_len >> 1);
    (void)gen.n_inverse;

    auto up = [](Complex64 **d, std::vector<Complex64> &h) {
        CU(cudaMalloc(d, h.size() * sizeof(Complex64)));
        CU(cudaMemcpy(*d, h.data(), h.size() * sizeof(Complex64), cudaMemcpyHostToDevice));
    };
    up(&d_root_fwd, rf);
    up(&d_root_inv, ri);
    up(&d_twist, tw);
    up(&d_untwist, ut);

    const size_t pb = (size_t)n_batch * padded * sizeof(Data64); // = n_batch*(n/2) complex
    CU(cudaMalloc(&d_buf_AB, 2 * pb));
    d_buf_A = d_buf_AB;
    d_buf_B = d_buf_AB + (size_t)n_batch * padded;
    CU(cudaMalloc(&d_real, (size_t)2 * n_batch * fft_len * sizeof(double)));
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    int n_tiles_max = (padded + CARRY_TILE - 1) / CARRY_TILE;
    CU(cudaMalloc(&d_tile_carry, (size_t)n_batch * n_tiles_max * sizeof(Data64)));
    CU(cudaMalloc(&d_first_tile, (size_t)n_batch * sizeof(int)));
#endif
}

FftFFNTBatch::~FftFFNTBatch()
{
    cudaFree(d_root_fwd);
    cudaFree(d_root_inv);
    cudaFree(d_twist);
    cudaFree(d_untwist);
    cudaFree(d_buf_AB);
    cudaFree(d_real);
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    cudaFree(d_tile_carry);
    cudaFree(d_first_tile);
#endif
}

// ── FFNT ──────────────────────────────────────────────────────────────────────

void FftFFNTBatch::run_ffnt(double *rbuf, Data64 *tbuf, bool fwd, int batch, cudaStream_t s)
{
    fft_configuration<Float64> cfg{};
    cfg.n_power = logn;
    cfg.fft_type = fwd ? FORWARD : INVERSE;
    cfg.zero_padding = false;
    cfg.stream = s;
    if (!fwd)
        cfg.mod_inverse = Complex64(n_inv, 0.0);
    GPU_FFNT(rbuf, reinterpret_cast<Complex64 *>(tbuf),
             fwd ? d_twist : d_untwist, fwd ? d_root_fwd : d_root_inv, cfg, batch, false);
}

void FftFFNTBatch::ntt_A(const LimbT *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_real, d_src, n_src, fft_len, n_batch);
    run_ffnt(d_real, d_buf_A, /*fwd=*/true, n_batch, s);
}

void FftFFNTBatch::ntt_B(const LimbT *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    double *rB = d_real + (size_t)n_batch * fft_len;
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(rB, d_src, n_src, fft_len, n_batch);
    run_ffnt(rB, d_buf_B, /*fwd=*/true, n_batch, s);
}

void FftFFNTBatch::ntt_AB(const LimbT *d_srcA, const LimbT *d_srcB, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    double *rB = d_real + (size_t)n_batch * fft_len;
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(d_real, d_srcA, n_src, fft_len, n_batch);
    load_real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(rB, d_srcB, n_src, fft_len, n_batch);
    run_ffnt(d_real, d_buf_A, /*fwd=*/true, 2 * n_batch, s); // A,B contiguous (real and temp)
}

void FftFFNTBatch::fwd_A(cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    real2real<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_real, reinterpret_cast<const double *>(d_buf_A), fft_len, padded, n_batch);
    run_ffnt(d_real, d_buf_A, /*fwd=*/true, n_batch, s);
}

void FftFFNTBatch::pmul(cudaStream_t s)
{
    int total = n_batch * (fft_len / 2); constexpr int thr = MR_THR_PMUL;
    cmul_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(reinterpret_cast<Complex64 *>(d_buf_A), reinterpret_cast<Complex64 *>(d_buf_B), total);
}

void FftFFNTBatch::psq(cudaStream_t s)
{
    int total = n_batch * (fft_len / 2); constexpr int thr = MR_THR_PMUL;
    csq_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(reinterpret_cast<Complex64 *>(d_buf_A), total);
}

void FftFFNTBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    int total = n_batch * (fft_len / 2); constexpr int thr = MR_THR_PMUL;
    cmul_kernel<<<(total + thr - 1) / thr, thr, 0, s>>>(reinterpret_cast<Complex64 *>(d_buf_A), reinterpret_cast<const Complex64 *>(d_ext), total);
}

void FftFFNTBatch::intt_A(cudaStream_t s)
{
    // Inverse FFNT leaves the (un-normalized, scaled) real coefficients in d_real,
    // which IS raw_coeffs() — the carry layer reads them directly as double.
    // No round_scatter: the double→int rounding/clamp now happens inside limb_ld
    // at the carry boundary (exact, since coefficients are < 2^52).
    run_ffnt(d_real, d_buf_A, /*fwd=*/false, n_batch, s);
}

void FftFFNTBatch::pmul_and_intt(cudaStream_t s) { pmul(s); intt_A(s); }
void FftFFNTBatch::psq_and_intt(cudaStream_t s) { psq(s); intt_A(s); }
void FftFFNTBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s) { pmul_ext(d_ext, s); intt_A(s); }

void FftFFNTBatch::schoolbook_mul(const LimbT *d_A, const LimbT *d_B, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL; unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        reinterpret_cast<double *>(d_real), d_A, d_B, n_src, fft_len, padded, n_batch);
}

void FftFFNTBatch::schoolbook_sq(const LimbT *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL; unsigned bx = (unsigned)(fft_len + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        reinterpret_cast<double *>(d_real), d_A, n_src, fft_len, padded, n_batch);
}
