/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/kernels/forward_core.cuh
 * ROLE   ForwardCore_Fused — first pass, gather fused into its load
 *
 * HOW    Handles the first OIC levels for n_power >= 10. The Op template
 *        applies the gather (zero-pad, shifted window, or two-source) on
 *        the global load itself, so the operand is never materialized in a
 *        separate buffer.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from ntt_merge_ntt_fused.cu. Stays a header
 *               because the kernel is a template the dispatcher
 *               instantiates.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// ForwardCore — fused first-pass kernel for n_power >= 10
// ─────────────────────────────────────────────────────────────────────────────

// OIC == outer_iteration_count, the butterfly-stage trip count; SI ==
// shared_index.  Templating both makes `loops`, the `offset / (1 << (OIC - 1))`
// divisor and every per-iteration shift amount compile-time constants — the
// `((x >> t_) << t_)` pairs collapse into single masked ops.
template <typename Op, int OIC, int SI, bool RPC>
__global__ static void ForwardCore_Fused(
    Data64 *polynomial_out,
    const Root64 *__restrict__ root_of_unity_table,
    const Data64 *__restrict__ root_shoup_table,
    Modulus64 modulus, int logm,
    int N_power, bool not_last_kernel,
    Op op)
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

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
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

    // Fused load: op gathers from the unpadded source and zero-fills the tail
    shared_memory[shared_addresss] = op[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = op[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;

    // One butterfly stage; the loop bodies below differ only in where the
    // __syncthreads() sits, so the stage itself lives in a lambda.
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
        // Last kernel: the final 6 stages fit inside a warp, so they need no
        // barrier between them (upstream's split).
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
    if (!not_last_kernel)
    {
        out0 = lazy_final(out0, modulus_reg.value);
        out1 = lazy_final(out1, modulus_reg.value);
    }
#endif
    polynomial_out[global_addresss] = out0;
    polynomial_out[global_addresss + offset] = out1;
}
