// ops/shift/shift.cu — kernels and launchers for limb shift/extract.
// Pure limb-value moves (no arithmetic) — templated on the limb storage type T
// (= LimbT): the stored representation (double or Data64) is copied verbatim.
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

    template <typename T>
    __global__ void extract_low_k(T *__restrict__ dst, const T *__restrict__ src,
                                  int n_low, int padded, int n_sum, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= padded)
            return;
        dst[(size_t)cand * padded + j] = (j < n_low) ? src[(size_t)cand * n_sum + j] : (T)0;
    }
}

namespace ops
{
    void shift_right(LimbT *dst, const LimbT *src, int offset,
                     int n_out, int n_src, int n_batch, int thr, cudaStream_t s)
    {
        dim3 g((unsigned)(n_out + thr - 1) / thr, (unsigned)n_batch);
        shift_right_k<LimbT><<<g, thr, 0, s>>>(dst, src, offset, n_out, n_src, n_batch);
    }

    void shift_right_var(LimbT *dst, const LimbT *src, const int *bark, int delta,
                         int n_out, int n_src, int n_batch, int thr, cudaStream_t s)
    {
        dim3 g((unsigned)(n_out + thr - 1) / thr, (unsigned)n_batch);
        shift_right_var_k<LimbT><<<g, thr, 0, s>>>(dst, src, bark, delta, n_out, n_src, n_batch);
    }

    void extract_low(LimbT *dst, const LimbT *src, int n_low, int padded,
                     int n_sum, int n_batch, int thr, cudaStream_t s)
    {
        dim3 g((unsigned)(padded + thr - 1) / thr, (unsigned)n_batch);
        extract_low_k<LimbT><<<g, thr, 0, s>>>(dst, src, n_low, padded, n_sum, n_batch);
    }
}
