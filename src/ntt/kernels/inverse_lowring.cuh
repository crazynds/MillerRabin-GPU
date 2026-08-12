/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/kernels/inverse_lowring.cuh
 * ROLE   InverseCoreLowRing_Fused — whole inverse transform in one kernel
 *
 * HOW    The n_power < 11 counterpart of the forward low-ring kernel: every
 *        level plus the 1/n scaling in a single block, no global round
 *        trip.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from ntt_merge_intt_fused.cu. Stays a header
 *               because the kernel is a template the dispatcher
 *               instantiates.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// InverseCoreLowRing — fused first (and only) kernel for n_power < 11
// ─────────────────────────────────────────────────────────────────────────────

// N_POWER is a template parameter so that `loops`, `half_n`, `m` and every shift
// become compile-time constants: the #pragma unroll below then fully unrolls and
// the address arithmetic collapses to immediates.  Only n_power < 11 reaches this
// kernel, so 10 instantiations cover every case (see launch_low_ring_fused).
template <typename Op, int N_POWER, bool RPC>
__global__ static void InverseCoreLowRing_Fused(
    Data64 *polynomial_out,
    const Root64 *__restrict__ inverse_root_of_unity_table,
    const Data64 *__restrict__ root_shoup_table,
    Modulus64 modulus,
    Ninverse64 n_inverse, Data64 n_inv_shoup, int total_batch,
    Op op)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int batch_index = (block_x * blockDim.y) + idx_y;
    const bool active_batch = (batch_index < total_batch);
    const int batch_index_safe = active_batch ? batch_index : 0;

    extern __shared__ char shared_memory_typed[];
    Data64 *shared_memory = reinterpret_cast<Data64 *>(shared_memory_typed);

    const Modulus64 modulus_reg = modulus;

    int t_2 = 0;
    int t_ = 0;
    const int offset = idx_y << N_POWER;
    constexpr int loops = N_POWER;
    int m = (int)1 << (N_POWER - 1);

    constexpr int half_n = 1 << (N_POWER - 1);
    const int row_base = idx_y << N_POWER;
    const int shared_address0 = row_base + idx_x;
    const int shared_address1 = shared_address0 + half_n;
    location_t global_address0 = idx_x + (location_t)(batch_index_safe << N_POWER);
    location_t global_address1 = global_address0 + half_n;

    // Fused load: op holds the input pointer(s) and applies the multiply
    shared_memory[shared_address0] = op[global_address0];
    shared_memory[shared_address1] = op[global_address1];

    int shared_addresss = idx_x;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    __syncthreads();

#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        int group_in_shared_address = in_shared_address + offset;
        if constexpr (RPC)
            current_root_index = (idx_x >> t_2);
        else
            current_root_index = m + (idx_x >> t_2);

        GentlemanSandeUnitX(shared_memory[group_in_shared_address],
                           shared_memory[group_in_shared_address + t],
                           inverse_root_of_unity_table, root_shoup_table,
                           current_root_index, modulus_reg);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        if constexpr (!RPC)
            m >>= 1;
        if (lp + 1 < loops)
            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        __syncthreads();
    }

    if (active_batch)
    {
        Data64 out0 = ninv_mul(shared_memory[shared_address0], n_inverse, n_inv_shoup, modulus_reg);
        Data64 out1 = ninv_mul(shared_memory[shared_address1], n_inverse, n_inv_shoup, modulus_reg);

        polynomial_out[global_address0] = out0;
        polynomial_out[global_address1] = out1;
    }
}
