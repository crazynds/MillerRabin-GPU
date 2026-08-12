/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/mr/mr_kernels.cu
 * ROLE   the two small kernels of the Miller-Rabin loop
 *
 * HOW    select_window_kernel gathers the power-table entry addressed by
 *        the current exponent window; check_equals_kernel compares a
 *        residue against a reference. Both are one short kernel each, so
 *        they share a file.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from mr/mr_internals.cu (524 lines).
 * ───────────────────────────────────────────────────────────────────────────── */
#include "mr/mr_internals.cuh"
#include "ops/mul/multiplier.cuh"
#include "util/cuda_check.cuh"

// ── Kernel: selects table[w] for each candidate given a window of WINDOW_BITS bits ──

__global__ void select_window_kernel(
    LimbT *__restrict__ d_out,
    const LimbT *__restrict__ d_table,
    const Data64 *__restrict__ d_exp,
    int msb_pos, int k,
    int n_limbs, int n_total)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_total || j >= n_limbs)
        return;

    int w = 0;
    for (int b = 0; b < k; b++)
    {
        int bp = msb_pos - b;
        if (bp >= 0)
        {
            int li = bp / LIMB_BITS;
            int bit = bp % LIMB_BITS;
            if ((d_exp[(size_t)t * n_limbs + li] >> bit) & 1)
                w |= (1 << (k - 1 - b));
        }
    }

    d_out[(size_t)t * n_limbs + j] =
        d_table[(size_t)w * n_total * n_limbs + (size_t)t * n_limbs + j];
}

// ── Kernel: checks whether d_r == d_ref for each candidate in d_alive ─────────────

__global__ void check_equals_kernel(
    const LimbT *__restrict__ d_r,
    const LimbT *__restrict__ d_ref,
    uint8_t *__restrict__ d_alive,
    int n_limbs, int n_total)
{
    int t = blockIdx.x;
    if (t >= n_total || d_alive[t] != 1)
        return;

    __shared__ int match;
    if (threadIdx.x == 0)
        match = 1;
    __syncthreads();

    const LimbT *rv = d_r + (size_t)t * n_limbs;
    const LimbT *ref = d_ref + (size_t)t * n_limbs;
    for (int j = (int)threadIdx.x; j < n_limbs; j += (int)blockDim.x)
        if (rv[j] != ref[j])
            atomicAnd(&match, 0);
    __syncthreads();

    if (threadIdx.x == 0 && match)
        d_alive[t] = 2;
}
