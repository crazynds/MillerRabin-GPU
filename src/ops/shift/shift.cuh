// ops/shift/shift.cuh — Limb shift/extract (base 2^LIMB_BITS), batched.
// Operations agnostic to the reduction algorithm; host launchers in shift.cu.
#pragma once

#include "ops/mul/multiplier.cuh" // Data64, LIMB_BITS (via selected backend)
#include <cuda_runtime.h>

namespace ops
{
    // dst[cand*n_out + j] = src[cand*n_src + j + offset] (0 outside the range).
    // Right shift by `offset` limbs, the same for all candidates.
    void shift_right(LimbT *dst, const LimbT *src, int offset,
                     int n_out, int n_src, int n_batch, cudaStream_t s);

    // Same, but offset = bark[cand] + delta (per-candidate).
    void shift_right_var(LimbT *dst, const LimbT *src, const int *bark, int delta,
                         int n_out, int n_src, int n_batch, cudaStream_t s);

    // dst[cand*padded + j] = (j < n_low) ? src[cand*n_sum + j] : 0 — extracts low limbs.
    void extract_low(LimbT *dst, const LimbT *src, int n_low, int padded,
                     int n_sum, int n_batch, cudaStream_t s);
}
