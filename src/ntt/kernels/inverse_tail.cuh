/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/kernels/inverse_tail.cuh
 * ROLE   InverseCore_Tail — every inverse pass after the first
 *
 * HOW    In-place like its forward twin; the last pass also multiplies by
 *        1/n.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from ntt_merge_intt_fused.cu. Stays a header
 *               because the kernel is a template the dispatcher
 *               instantiates.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Remaining-pass kernels (identical to GPU-NTT InverseCore / InverseCore_,
// copied here because __global__ template symbols are not exported by the lib).
// ─────────────────────────────────────────────────────────────────────────────

template <int OIC, int SI, bool RPC>
__global__ static void InverseCore_Tail(
    Data64 *polynomial_in, Data64 *polynomial_out,
    const Root64 *__restrict__ inverse_root_of_unity_table,
    const Data64 *__restrict__ root_shoup_table,
    Modulus64 modulus, int logm, int k,
    int N_power, Ninverse64 n_inverse, Data64 n_inv_shoup,
    bool last_kernel)
{
    constexpr int outer_iteration_count = OIC;
    constexpr int shared_index = SI;
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    extern __shared__ char shared_memory_typed[];
    Data64 *shared_memory = reinterpret_cast<Data64 *>(shared_memory_typed);

    const Modulus64 modulus_reg = modulus;
    const Ninverse64 n_inverse_reg = n_inverse;

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    int t_ = (shared_index + 1) - outer_iteration_count;
    constexpr int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) +
        (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) +
        (location_t)(block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;

#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if constexpr (RPC)
            current_root_index = (omega_addresss >> t_2);
        else
            current_root_index = m + (omega_addresss >> t_2);

        GentlemanSandeUnitX(shared_memory[in_shared_address],
                           shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table, root_shoup_table,
                           current_root_index, modulus_reg);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        if constexpr (!RPC)
            m >>= 1;
        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        Data64 out0 = ninv_mul(shared_memory[shared_addresss], n_inverse_reg, n_inv_shoup, modulus_reg);
        Data64 out1 = ninv_mul(shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse_reg, n_inv_shoup, modulus_reg);
        polynomial_out[global_addresss] = out0;
        polynomial_out[global_addresss + offset] = out1;
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}
