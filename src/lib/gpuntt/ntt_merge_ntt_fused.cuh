#pragma once
// Custom forward-NTT variants that fold the zero-padded gather into the first
// kernel's global-memory load, eliminating the separate load_padded_batch pass
// (one full read + one full write over n_batch * padded words per operand).
//
// This is the mirror image of ntt_merge_intt_fused.cuh, which folds the
// pointwise multiply into the INTT's first load.
//
// Only the PerPolynomial / Data64 (unsigned) path with ReductionPolynomial
// X_N_minus is implemented — the only path used by this project.  The dispatch
// throws if given X_N_plus.
//
// Source layout: src is [batch][n_src], the transform buffer is
// [batch][1 << cfg.n_power]; coefficients beyond n_src read as zero.

#include "gpuntt/ntt_merge/ntt.cuh"

using namespace gpuntt;

// Fused: device_out[c][j] = NTT( c < batch ? src[c][j] : 0 )
__host__ void GPU_NTT_ZeroPadLoad(
    Data64 *device_out,
    const Data64 *src, int n_src,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Fused two-source variant for the A|B contiguous layout: candidates
// [0, n_batch) gather from srcA, candidates [n_batch, 2*n_batch) from srcB.
// batch_size must be 2 * n_batch.
__host__ void GPU_NTT_ZeroPadLoad2(
    Data64 *device_out,
    const Data64 *srcA, const Data64 *srcB, int n_src, int n_batch,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);
