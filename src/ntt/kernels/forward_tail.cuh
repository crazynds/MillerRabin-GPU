/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/kernels/forward_tail.cuh
 * ROLE   ForwardCore_Tail — every pass after the first
 *
 * HOW    Reads and writes the transform buffer in place: each block owns
 *        exactly the pair of addresses it loads, so no block touches
 *        another block data. That is what makes the in-place entry point
 *        safe.
 *
 * NOTE   Do NOT try to fix the shared-memory bank conflicts here. Two attempts
 *        are recorded and rejected: additive padding (4.2% SLOWER) and an XOR
 *        swizzle (3.5% SLOWER, implementation since removed). The swizzle removed
 *        ~240x of the conflicts and 28% of the shared wavefronts and was STILL
 *        slower, which settles the question: a conflict costs a replay inside
 *        the memory pipeline, not an instruction, so the LSU instruction count
 *        does not move — and at 87-90% occupancy those replays are already
 *        hidden, worth only ~0.6 of ~21 stall-cycles per issue. The swizzle
 *        arithmetic, by contrast, is +13% instructions on an issue-bound
 *        kernel. The 2.8-vs-2.0 wavefront ratio is a real measurement of an
 *        effect that does not matter. Reject anything justified by that ratio
 *        unless it also cuts instruction count. Full derivation and numbers:
 *        docs/ntt-bank-conflicts.md.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from ntt_merge_ntt_fused.cu. Stays a header
 *               because the kernel is a template the dispatcher
 *               instantiates.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Remaining-pass kernels (identical to GPU-NTT ForwardCore / ForwardCore_,
// copied here because __global__ template symbols are not exported by the lib).
// ─────────────────────────────────────────────────────────────────────────────

// TRANSPOSED == false → ForwardCore, true → ForwardCore_ (block_x ↔ block_y).
template <int OIC, int SI, bool RPC, bool TRANSPOSED>
__global__ static void ForwardCore_Tail(
    Data64 *polynomial_in, Data64 *polynomial_out,
    const Root64 *__restrict__ root_of_unity_table,
    const Data64 *__restrict__ root_shoup_table,
    Modulus64 modulus, int logm,
    int N_power, bool not_last_kernel)
{
    constexpr int outer_iteration_count = OIC;
    constexpr int shared_index = SI;
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    // ForwardCore_ swaps which block dimension walks the coefficient axis.
    const int blk_lin = TRANSPOSED ? block_y : block_x;
    const int blk_grp = TRANSPOSED ? block_x : block_y;

    extern __shared__ char shared_memory_typed[];
    Data64 *shared_memory = reinterpret_cast<Data64 *>(shared_memory_typed);

    const Modulus64 modulus_reg = modulus;

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * blk_lin) +
        (location_t)(2 * blk_grp * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * blk_lin) +
        (location_t)(blk_grp * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;

    auto stage = [&]()
    {
        if constexpr (RPC)
            current_root_index = (omega_addresss >> t_2);
        else
            current_root_index = m + (omega_addresss >> t_2);

        CooleyTukeyUnitX(shared_memory[in_shared_address],
                        shared_memory[in_shared_address + t],
                        root_of_unity_table, root_shoup_table,
                        current_root_index, modulus_reg);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        if constexpr (!RPC)
            m <<= 1;
        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    };

    if (not_last_kernel)
    {
#pragma unroll
        for (int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            stage();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for (int lp = 0; lp < (shared_index - 5); lp++)
        {
            __syncthreads();
            stage();
        }
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < 6; lp++)
            stage();
        __syncthreads();
    }

    Data64 out0 = shared_memory[shared_addresss];
    Data64 out1 = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
#ifdef MR_NTT_LAZY
    // Coefficients ride between passes in [0,4p); the final pass hands the
    // pointwise multiply canonical residues.
    if (!not_last_kernel)
    {
        out0 = lazy_final(out0, modulus_reg.value);
        out1 = lazy_final(out1, modulus_reg.value);
    }
#endif
    polynomial_out[global_addresss] = out0;
    polynomial_out[global_addresss + offset] = out1;
}
