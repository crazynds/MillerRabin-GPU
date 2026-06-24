#pragma once
// Custom fused INTT variants that fold the pointwise multiply/square into the
// first kernel's global-memory load, eliminating one full read+write pass over
// the coefficient array.
//
// Only the PerPolynomial / Data64 (unsigned) path is implemented — that is the
// only path used by this project.  The API mirrors GPU_INTT_Inplace from
// GPU-NTT; callers just swap the call site in pmul_and_intt / psq_and_intt.

#include "gpuntt/ntt_merge/ntt.cuh"

using namespace gpuntt;

// Fused: a[i] = INTT(a[i] * b[i] mod p)
__host__ void GPU_INTT_Inplace_PreMul(
    Data64 *device_inout,
    const Data64 *b,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);

// Fused: a[i] = INTT(a[i]^2 mod p)
__host__ void GPU_INTT_Inplace_PreSq(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size);
