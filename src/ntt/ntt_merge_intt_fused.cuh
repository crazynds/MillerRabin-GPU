/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/ntt_merge_intt_fused.cuh
 * ROLE   public entry points of the inverse NTT
 *
 * HOW    One entry per pointwise shape: multiply by an external spectrum,
 *        square, or nothing. The pointwise stage rides in the first kernel
 *        load, so it never costs its own pass over the buffer.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "gpuntt/ntt_merge/ntt.cuh"

using namespace gpuntt;

// Fused: a[i] = INTT(a[i] * b[i] mod p)
__host__ void GPU_INTT_Inplace_PreMul(
    Data64 *device_inout,
    const Data64 *b,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Fused: a[i] = INTT(a[i]^2 mod p)
__host__ void GPU_INTT_Inplace_PreSq(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Plain in-place inverse NTT through the local (Shoup-enabled) kernels — the
// drop-in replacement for the library's GPU_INTT_Inplace.
__host__ void GPU_INTT_Inplace_Shoup(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);
