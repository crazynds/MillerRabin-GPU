// Fused forward-NTT variants: fold the zero-padded gather into the first
// kernel's global-memory load.  See ntt_merge_ntt_fused.cuh for the public API.
//
// Kernel layout mirrors the GPU-NTT merge backend (PerPolynomial, unsigned):
//
//   n_power <  10 → single ForwardCoreLowRing kernel (entire NTT in one shot)
//   n_power 10-24 → multiple ForwardCore passes (standard kernel)
//   n_power >= 25 → same, except the LAST pass uses ForwardCore_ (transposed)
//
// Note the asymmetry with the INTT: there the transposed kernel is the *first*
// pass, so it had to be fused too.  Here pass 0 is always the standard layout,
// so only ForwardCore (and the low-ring kernel) need a fused variant.
//
// Source reference: GPU-NTT ntt_merge/ntt.cu (Apache-2.0, Alisah Özcan)

#include "ntt_merge_ntt_fused.cuh"
#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"

#include "helpers/const_dispatch.h"
#include "ntt_merge_kernel_bounds.h"

using namespace gpuntt;

#define GPUNTT_CHECK(expr)                                                                             \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t _e = (expr);                                                                       \
        if (_e != cudaSuccess)                                                                         \
            throw std::runtime_error(std::string("[fused_ntt] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Gather ops applied at the two global loads per thread.  `idx` is a linear
// index into the padded transform buffer, so (candidate, coefficient) is
// recovered with a shift/mask against the transform length.
// ─────────────────────────────────────────────────────────────────────────────

struct OpPad
{
    const Data64 *src;
    int n_src;
    int logp; // log2(transform length) == cfg.n_power

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        const location_t j = idx & (((location_t)1 << logp) - 1);
        if (j >= (location_t)n_src)
            return 0ULL;
        return src[(idx >> logp) * (location_t)n_src + j];
    }
};

// Candidates [0, n_batch) gather from srcA, the rest from srcB — matches the
// A|B contiguous buffer that ntt_AB transforms in a single launch.
struct OpPad2
{
    const Data64 *srcA;
    const Data64 *srcB;
    int n_src;
    int n_batch;
    int logp;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        const location_t j = idx & (((location_t)1 << logp) - 1);
        if (j >= (location_t)n_src)
            return 0ULL;
        location_t cand = idx >> logp;
        const Data64 *src = srcA;
        if (cand >= (location_t)n_batch)
        {
            src = srcB;
            cand -= (location_t)n_batch;
        }
        return src[cand * (location_t)n_src + j];
    }
};

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

        CooleyTukeyUnit(shared_memory[group_in_shared_address],
                        shared_memory[group_in_shared_address + t],
                        root_of_unity_table[current_root_index],
                        modulus_reg);

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
        polynomial_out[global_address0] = shared_memory[shared_address0];
        polynomial_out[global_address1] = shared_memory[shared_address1];
    }
}

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

        CooleyTukeyUnit(shared_memory[in_shared_address],
                        shared_memory[in_shared_address + t],
                        root_of_unity_table[current_root_index],
                        modulus_reg);

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

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

// ─────────────────────────────────────────────────────────────────────────────
// Remaining-pass kernels (identical to GPU-NTT ForwardCore / ForwardCore_,
// copied here because __global__ template symbols are not exported by the lib).
// ─────────────────────────────────────────────────────────────────────────────

// TRANSPOSED == false → ForwardCore, true → ForwardCore_ (block_x ↔ block_y).
template <int OIC, int SI, bool RPC, bool TRANSPOSED>
__global__ static void ForwardCore_Tail(
    Data64 *polynomial_in, Data64 *polynomial_out,
    const Root64 *__restrict__ root_of_unity_table,
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

        CooleyTukeyUnit(shared_memory[in_shared_address],
                        shared_memory[in_shared_address + t],
                        root_of_unity_table[current_root_index],
                        modulus_reg);

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

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared dispatch logic
// ─────────────────────────────────────────────────────────────────────────────

// This project always runs X_N_minus, so only that specialisation is
// instantiated — building both would double an already large template fan-out
// (Op x OIC x SI) for a branch that never executes.
static constexpr bool FUSED_RPC = true;

template <typename Op>
static void dispatch_ntt_fused(
    Data64 *device_out,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size,
    Op op)
{
    if (cfg.ntt_layout != PerPolynomial)
        throw std::runtime_error("[fused_ntt] only the PerPolynomial layout is supported");
    if (cfg.reduction_poly != ReductionPolynomial::X_N_minus)
        throw std::runtime_error("[fused_ntt] only ReductionPolynomial::X_N_minus is supported");

    auto kernel_params = CreateForwardNTTKernel<Data64>();
    const bool low_ring = (cfg.n_power < 10);
    const bool std_kernel = (cfg.n_power < 25);

    auto dispatch_pass = [&](const KernelConfig &p, auto &&launch)
    {
        dispatch_const<1, MAX_OUTER_ITERATION_COUNT>(
            p.outer_iteration_count, [&](auto oic)
            { dispatch_const<MIN_SHARED_INDEX, MAX_SHARED_INDEX>(
                  p.shared_index, [&](auto si)
                  { launch(oic, si); }); });
        GPUNTT_CHECK(cudaGetLastError());
    };

    // Every pass after the first is the plain (non-fused) tail kernel.
    auto launch_tail = [&](const KernelConfig &p, bool transposed)
    {
        dispatch_pass(p, [&](auto oic, auto si)
                      {
            auto launch_with = [&](auto tr)
            {
                ForwardCore_Tail<oic.value, si.value, FUSED_RPC, tr.value><<<
                    dim3(p.griddim_x, p.griddim_y, batch_size),
                    dim3(p.blockdim_x, p.blockdim_y),
                    p.shared_memory, cfg.stream>>>(
                    device_out, device_out,
                    root_of_unity_table, modulus,
                    p.logm, cfg.n_power, p.not_last_kernel);
            };
            if (transposed)
                launch_with(std::integral_constant<bool, true>{});
            else
                launch_with(std::integral_constant<bool, false>{}); });
    };

    if (low_ring)
    {
        // shared_index is always n_power - 1 here, so n_power is the only
        // constant the kernel needs.
        auto &p = kernel_params[cfg.n_power][0];
        int grid_x = (batch_size + p.blockdim_y - 1) / p.blockdim_y;
        dispatch_const<1, MAX_FWD_LOW_RING_N_POWER>(
            cfg.n_power, [&](auto np)
            { ForwardCoreLowRing_Fused<Op, np.value, FUSED_RPC><<<
                  dim3(grid_x, 1, 1),
                  dim3(p.blockdim_x, p.blockdim_y),
                  p.shared_memory, cfg.stream>>>(
                  device_out, root_of_unity_table, modulus,
                  batch_size, op); });
        GPUNTT_CHECK(cudaGetLastError());
        return;
    }

    // Pass 0 is the fused kernel; it is the standard block layout on every
    // n_power (unlike the INTT, only the *last* forward pass is transposed).
    {
        auto &p = kernel_params[cfg.n_power][0];
        dispatch_pass(p, [&](auto oic, auto si)
                      { ForwardCore_Fused<Op, oic.value, si.value, FUSED_RPC><<<
                            dim3(p.griddim_x, p.griddim_y, batch_size),
                            dim3(p.blockdim_x, p.blockdim_y),
                            p.shared_memory, cfg.stream>>>(
                            device_out,
                            root_of_unity_table, modulus,
                            p.logm, cfg.n_power, p.not_last_kernel, op); });
    }

    const int n_pass = (int)kernel_params[cfg.n_power].size();
    // For n_power >= 25 the final pass uses the transposed block layout.
    const int n_std = std_kernel ? n_pass : n_pass - 1;
    for (int i = 1; i < n_std; i++)
        launch_tail(kernel_params[cfg.n_power][i], false);
    if (!std_kernel)
        launch_tail(kernel_params[cfg.n_power][n_pass - 1], true);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

__host__ void GPU_NTT_ZeroPadLoad(
    Data64 *device_out,
    const Data64 *src, int n_src,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpPad op{src, n_src, cfg.n_power};
    dispatch_ntt_fused(device_out, root_of_unity_table, modulus, cfg, batch_size, op);
}

__host__ void GPU_NTT_ZeroPadLoad2(
    Data64 *device_out,
    const Data64 *srcA, const Data64 *srcB, int n_src, int n_batch,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpPad2 op{srcA, srcB, n_src, n_batch, cfg.n_power};
    dispatch_ntt_fused(device_out, root_of_unity_table, modulus, cfg, batch_size, op);
}
