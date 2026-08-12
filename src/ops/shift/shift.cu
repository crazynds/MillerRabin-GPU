/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/shift/shift.cu
 * ROLE   batched limb shift and extract
 *
 * HOW    Pure limb moves with no arithmetic, so the stored representation
 *        is copied verbatim. extract_low is the exception: it writes the
 *        transform buffer, whose word type differs from the limb type under
 *        MR_LIMB32, so it carries a split source/destination type.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "ops/shift/shift.cuh"

namespace
{
    template <typename T>
    __global__ void shift_right_k(T *__restrict__ dst, const T *__restrict__ src,
                                  int offset, int n_out, int n_src, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= n_out)
            return;
        int sidx = j + offset;
        dst[(size_t)cand * n_out + j] = (sidx < n_src) ? src[(size_t)cand * n_src + sidx] : (T)0;
    }

    template <typename T>
    __global__ void shift_right_var_k(T *__restrict__ dst, const T *__restrict__ src,
                                      const int *__restrict__ bark, int delta,
                                      int n_out, int n_src, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= n_out)
            return;
        int off = bark[cand] + delta;
        int sidx = j + off;
        dst[(size_t)cand * n_out + j] =
            (sidx >= 0 && sidx < n_src) ? src[(size_t)cand * n_src + sidx] : (T)0;
    }

    // dst is the transform buffer (RawT), src a limb array — the two widths differ
    // under MR_LIMB32, so this is the one shift kernel with a split type.
    template <typename TDst, typename TSrc>
    __global__ void extract_low_k(TDst *__restrict__ dst, const TSrc *__restrict__ src,
                                  int n_low, int padded, int n_sum, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= padded)
            return;
        dst[(size_t)cand * padded + j] =
            (j < n_low) ? (TDst)limb_ld(src[(size_t)cand * n_sum + j]) : (TDst)0;
    }
}

namespace ops
{
    void shift_right(LimbT *dst, const LimbT *src, int offset,
                     int n_out, int n_src, int n_batch, cudaStream_t s)
    {
        const int thr = MR_THR_COPY;
        dim3 g((unsigned)(n_out + thr - 1) / thr, (unsigned)n_batch);
        shift_right_k<LimbT><<<g, thr, 0, s>>>(dst, src, offset, n_out, n_src, n_batch);
    }

    void shift_right_var(LimbT *dst, const LimbT *src, const int *bark, int delta,
                         int n_out, int n_src, int n_batch, cudaStream_t s)
    {
        const int thr = MR_THR_COPY;
        dim3 g((unsigned)(n_out + thr - 1) / thr, (unsigned)n_batch);
        shift_right_var_k<LimbT><<<g, thr, 0, s>>>(dst, src, bark, delta, n_out, n_src, n_batch);
    }

    void extract_low(RawT *dst, const LimbT *src, int n_low, int padded,
                     int n_sum, int n_batch, cudaStream_t s)
    {
        const int thr = MR_THR_COPY;
        dim3 g((unsigned)(padded + thr - 1) / thr, (unsigned)n_batch);
        extract_low_k<RawT, LimbT><<<g, thr, 0, s>>>(dst, src, n_low, padded, n_sum, n_batch);
    }
}
