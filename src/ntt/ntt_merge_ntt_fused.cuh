/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/ntt_merge_ntt_fused.cuh
 * ROLE   public entry points of the forward NTT
 *
 * HOW    One entry per gather shape: zero-pad, two-source (the A|B pair in
 *        one launch), per-candidate shifted window, and plain in-place. All
 *        of them fold the gather into the first kernel load instead of
 *        materializing the padded operand first.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "gpuntt/ntt_merge/ntt.cuh"
#include "ops/limb_storage.cuh"

using namespace gpuntt;

// Fused: device_out[c][j] = NTT( c < batch ? src[c][j] : 0 )
__host__ void GPU_NTT_ZeroPadLoad(
    Data64 *device_out,
    const LimbT *src, int n_src,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Fused two-source variant for the A|B contiguous layout: candidates
// [0, n_batch) gather from srcA, candidates [n_batch, 2*n_batch) from srcB.
// batch_size must be 2 * n_batch.
__host__ void GPU_NTT_ZeroPadLoad2(
    Data64 *device_out,
    const LimbT *srcA, const LimbT *srcB, int n_src, int n_batch,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Fused variable-shift gather: coefficient j of candidate `cand` reads
// src[cand*n_src + j + bark[cand] + delta] (zero outside), for j < n_out.
// Folds ops::shift_right_var into the forward transform.
__host__ void GPU_NTT_ShiftPadLoad(
    Data64 *device_out,
    const LimbT *src, const int *bark, int delta, int n_out, int n_src,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Plain in-place forward NTT through the local (Shoup-enabled) kernels — the
// drop-in replacement for the library's GPU_NTT_Inplace.
__host__ void GPU_NTT_Inplace_Shoup(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);
