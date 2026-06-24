// Fused INTT variants: fold pointwise multiply/square into the first kernel's
// global-memory load.  See ntt_merge_intt_fused.cuh for the public API.
//
// Kernel layout mirrors the GPU-NTT merge backend (PerPolynomial, unsigned):
//
//   n_power <  11 → single InverseCoreLowRing kernel (entire INTT in one shot)
//   n_power 11-24 → multiple InverseCore passes  (standard kernel)
//   n_power >= 25 → first pass uses InverseCore_ (transposed blocks), rest InverseCore
//
// Only the FIRST kernel of each path has its global-memory load replaced with the
// fused operation.  All subsequent passes are identical to the upstream kernels.
//
// Source reference: GPU-NTT ntt_merge/ntt.cu (Apache-2.0, Alisah Özcan)

#include "ntt_merge_intt_fused.cuh"
#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"

using namespace gpuntt;

#define GPUNTT_CHECK(expr)                                                                              \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t _e = (expr);                                                                        \
        if (_e != cudaSuccess)                                                                          \
            throw std::runtime_error(std::string("[fused_intt] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Helper: NTT-domain pointwise op applied at the two global loads per thread.
// Defined as device lambdas emulated via a tag-dispatch struct so the compiler
// can inline the operation without branching in the hot loop.
// ─────────────────────────────────────────────────────────────────────────────

struct OpMul
{
    const Data64 *a;
    const Data64 *b;
    Data64 modulus_val;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        return (Data64)((__uint128_t)a[idx] * b[idx] % (__uint128_t)modulus_val);
    }
};

struct OpSq
{
    const Data64 *a;
    Data64 modulus_val;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        return (Data64)((__uint128_t)a[idx] * a[idx] % (__uint128_t)modulus_val);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// InverseCoreLowRing — fused first (and only) kernel for n_power < 11
// ─────────────────────────────────────────────────────────────────────────────

template <typename Op>
__global__ static void InverseCoreLowRing_Fused(
    Data64 *polynomial_out,
    const Root64 *__restrict__ inverse_root_of_unity_table,
    Modulus64 modulus, int shared_index, int N_power,
    Ninverse64 n_inverse, bool reduction_poly_check, int total_batch,
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
    int offset = idx_y << N_power;
    int loops = N_power;
    int m = (int)1 << (N_power - 1);

    const int half_n = 1 << (N_power - 1);
    const int row_base = idx_y << N_power;
    const int shared_address0 = row_base + idx_x;
    const int shared_address1 = shared_address0 + half_n;
    location_t global_address0 = idx_x + (location_t)(batch_index_safe << N_power);
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
        if (reduction_poly_check)
        {
            current_root_index = (idx_x >> t_2);
        }
        else
        {
            current_root_index = m + (idx_x >> t_2);
        }

        GentlemanSandeUnit(shared_memory[group_in_shared_address],
                           shared_memory[group_in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index],
                           modulus_reg);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        if (lp + 1 < loops)
            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        __syncthreads();
    }
    __syncthreads();

    Data64 out0 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_address0], n_inverse, modulus_reg);
    Data64 out1 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_address1], n_inverse, modulus_reg);

    if (active_batch)
    {
        polynomial_out[global_address0] = out0;
        polynomial_out[global_address1] = out1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InverseCore — fused first-pass kernel for n_power 11-24
// ─────────────────────────────────────────────────────────────────────────────

template <typename Op>
__global__ static void InverseCore_Fused(
    Data64 *polynomial_out,
    const Root64 *__restrict__ inverse_root_of_unity_table,
    Modulus64 modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse64 n_inverse,
    bool last_kernel, bool reduction_poly_check,
    Op op)
{
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
    int loops = outer_iteration_count;
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

    // Fused load: op holds the input pointer(s) and applies the multiply
    shared_memory[shared_addresss] = op[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = op[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;

#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        {
            current_root_index = (omega_addresss >> t_2);
        }
        else
        {
            current_root_index = m + (omega_addresss >> t_2);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address],
                           shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index],
                           modulus_reg);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        Data64 out0 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
        Data64 out1 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse_reg, modulus_reg);
        polynomial_out[global_addresss] = out0;
        polynomial_out[global_addresss + offset] = out1;
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InverseCore_ — fused first-pass kernel for n_power >= 25 (transposed block layout)
// ─────────────────────────────────────────────────────────────────────────────

template <typename Op>
__global__ static void InverseCore__Fused(
    Data64 *polynomial_out,
    const Root64 *__restrict__ inverse_root_of_unity_table,
    Modulus64 modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse64 n_inverse,
    bool last_kernel, bool reduction_poly_check,
    Op op)
{
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
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    // Transposed: block_x ↔ block_y compared to InverseCore
    location_t global_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) +
        (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) +
        (location_t)(block_x * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Fused load: op holds the input pointer(s) and applies the multiply
    shared_memory[shared_addresss] = op[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] = op[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;

#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        {
            current_root_index = (omega_addresss >> t_2);
        }
        else
        {
            current_root_index = m + (omega_addresss >> t_2);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address],
                           shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index],
                           modulus_reg);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        Data64 out0 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
        Data64 out1 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse_reg, modulus_reg);
        polynomial_out[global_addresss] = out0;
        polynomial_out[global_addresss + offset] = out1;
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Remaining-pass kernels (identical to GPU-NTT InverseCore / InverseCore_,
// copied here because __global__ template symbols are not exported by the lib).
// ─────────────────────────────────────────────────────────────────────────────

__global__ static void InverseCore_Tail(
    Data64 *polynomial_in, Data64 *polynomial_out,
    const Root64 *__restrict__ inverse_root_of_unity_table,
    Modulus64 modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse64 n_inverse,
    bool last_kernel, bool reduction_poly_check)
{
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
    int loops = outer_iteration_count;
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
        if (reduction_poly_check)
        {
            current_root_index = (omega_addresss >> t_2);
        }
        else
        {
            current_root_index = m + (omega_addresss >> t_2);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address],
                           shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index],
                           modulus_reg);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        Data64 out0 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
        Data64 out1 = OPERATOR_GPU<Data64>::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse_reg, modulus_reg);
        polynomial_out[global_addresss] = out0;
        polynomial_out[global_addresss + offset] = out1;
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] = shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared dispatch logic
// ─────────────────────────────────────────────────────────────────────────────

template <typename Op>
static void dispatch_intt_fused(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size,
    Op op)
{
    auto kernel_params = CreateInverseNTTKernel<Data64>();
    const bool low_ring = (cfg.n_power < 11);
    const bool std_kernel = (cfg.n_power < 25);
    const bool reduction_poly_check =
        (cfg.reduction_poly == ReductionPolynomial::X_N_minus);

    if (low_ring)
    {
        auto &p = kernel_params[cfg.n_power][0];
        int grid_x = (batch_size + p.blockdim_y - 1) / p.blockdim_y;
        InverseCoreLowRing_Fused<<<
            dim3(grid_x, 1, 1),
            dim3(p.blockdim_x, p.blockdim_y),
            p.shared_memory, cfg.stream>>>(
            device_inout,
            root_of_unity_table, modulus,
            p.shared_index, cfg.n_power, cfg.mod_inverse,
            reduction_poly_check, batch_size, op);
        GPUNTT_CHECK(cudaGetLastError());
    }
    else if (std_kernel)
    {
        // Pass 0: fused first kernel
        {
            auto &p = kernel_params[cfg.n_power][0];
            InverseCore_Fused<<<
                dim3(p.griddim_x, p.griddim_y, batch_size),
                dim3(p.blockdim_x, p.blockdim_y),
                p.shared_memory, cfg.stream>>>(
                device_inout,
                root_of_unity_table, modulus,
                p.shared_index, p.logm, p.k, p.outer_iteration_count,
                cfg.n_power, cfg.mod_inverse,
                p.not_last_kernel, reduction_poly_check, op);
            GPUNTT_CHECK(cudaGetLastError());
        }
        // Remaining passes: plain tail kernel
        for (int i = 1; i < (int)kernel_params[cfg.n_power].size(); i++)
        {
            auto &p = kernel_params[cfg.n_power][i];
            InverseCore_Tail<<<
                dim3(p.griddim_x, p.griddim_y, batch_size),
                dim3(p.blockdim_x, p.blockdim_y),
                p.shared_memory, cfg.stream>>>(
                device_inout, device_inout,
                root_of_unity_table, modulus,
                p.shared_index, p.logm, p.k, p.outer_iteration_count,
                cfg.n_power, cfg.mod_inverse,
                p.not_last_kernel, reduction_poly_check);
            GPUNTT_CHECK(cudaGetLastError());
        }
    }
    else
    {
        // Pass 0: fused using transposed-block kernel (InverseCore_ equivalent)
        {
            auto &p = kernel_params[cfg.n_power][0];
            InverseCore__Fused<<<
                dim3(p.griddim_x, p.griddim_y, batch_size),
                dim3(p.blockdim_x, p.blockdim_y),
                p.shared_memory, cfg.stream>>>(
                device_inout,
                root_of_unity_table, modulus,
                p.shared_index, p.logm, p.k, p.outer_iteration_count,
                cfg.n_power, cfg.mod_inverse,
                p.not_last_kernel, reduction_poly_check, op);
            GPUNTT_CHECK(cudaGetLastError());
        }
        // Remaining passes
        for (int i = 1; i < (int)kernel_params[cfg.n_power].size(); i++)
        {
            auto &p = kernel_params[cfg.n_power][i];
            InverseCore_Tail<<<
                dim3(p.griddim_x, p.griddim_y, batch_size),
                dim3(p.blockdim_x, p.blockdim_y),
                p.shared_memory, cfg.stream>>>(
                device_inout, device_inout,
                root_of_unity_table, modulus,
                p.shared_index, p.logm, p.k, p.outer_iteration_count,
                cfg.n_power, cfg.mod_inverse,
                p.not_last_kernel, reduction_poly_check);
            GPUNTT_CHECK(cudaGetLastError());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

__host__ void GPU_INTT_Inplace_PreMul(
    Data64 *device_inout,
    const Data64 *b,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpMul op{device_inout, b, modulus.value};
    dispatch_intt_fused(device_inout, root_of_unity_table, modulus, cfg, batch_size, op);
}

__host__ void GPU_INTT_Inplace_PreSq(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpSq op{device_inout, modulus.value};
    dispatch_intt_fused(device_inout, root_of_unity_table, modulus, cfg, batch_size, op);
}
