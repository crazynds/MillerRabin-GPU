/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/kernels/forward_lowring.cuh
 * ROLE   ForwardCoreLowRing_Fused — whole transform in one kernel
 *
 * HOW    For n_power < 10 the entire forward transform fits one block, so
 *        every level runs in shared memory with no global round trip.
 *        N_POWER is a template parameter, so every shift and address
 *        becomes an immediate and the loop unrolls fully.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from ntt_merge_ntt_fused.cu. Stays a header
 *               because the kernel is a template the dispatcher
 *               instantiates.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// ForwardCoreLowRing — fused first (and only) kernel for n_power < 10
// ─────────────────────────────────────────────────────────────────────────────

// N_POWER is a template parameter so that `loops`, `half_n`, `t_` and every
// shift become compile-time constants: the #pragma unroll below then fully
// unrolls and the address arithmetic collapses to immediates.  RPC is the
// X_N_minus flag; templating it removes a predicate from every unrolled
// butterfly and (for RPC == true) kills the `m` chain entirely.
template <typename Op, int N_POWER, bool RPC>
__global__ static void ForwardCoreLowRing_Fused(
    Data64 *polynomial_out,
    const Root64 *__restrict__ root_of_unity_table,
    const Data64 *__restrict__ root_shoup_table,
    Modulus64 modulus, int total_batch,
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

    // The low-ring configs all have shared_index == N_POWER - 1, so t_ and the
    // trip count follow from N_POWER alone.
    int t_2 = N_POWER - 1;
    int t_ = N_POWER - 1;
    const int offset = idx_y << N_POWER;
    constexpr int loops = N_POWER;
    location_t m = 1;

    constexpr int half_n = 1 << (N_POWER - 1);
    const int row_base = idx_y << N_POWER;
    const int shared_address0 = row_base + idx_x;
    const int shared_address1 = shared_address0 + half_n;
    const location_t global_address0 = idx_x + (location_t)(batch_index_safe << N_POWER);
    const location_t global_address1 = global_address0 + half_n;
    const location_t omega_addresss = idx_x;

    // Fused load: op gathers from the unpadded source and zero-fills the tail
    shared_memory[shared_address0] = op[global_address0];
    shared_memory[shared_address1] = op[global_address1];

    const int shared_addresss = idx_x;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    __syncthreads();

#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        int group_in_shared_address = in_shared_address + offset;
        if constexpr (RPC)
            current_root_index = (omega_addresss >> t_2);
        else
            current_root_index = m + (omega_addresss >> t_2);

        CooleyTukeyUnitX(shared_memory[group_in_shared_address],
                        shared_memory[group_in_shared_address + t],
                        root_of_unity_table, root_shoup_table,
                        current_root_index, modulus_reg);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        if constexpr (!RPC)
            m <<= 1;
        if (lp + 1 < loops)
            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        __syncthreads();
    }

    if (active_batch)
    {
        Data64 out0 = shared_memory[shared_address0];
        Data64 out1 = shared_memory[shared_address1];
#ifdef MR_NTT_LAZY
        // This kernel is the whole transform, so it always reduces on the way out.
        out0 = lazy_final(out0, modulus_reg.value);
        out1 = lazy_final(out1, modulus_reg.value);
#endif
        polynomial_out[global_address0] = out0;
        polynomial_out[global_address1] = out1;
    }
}
