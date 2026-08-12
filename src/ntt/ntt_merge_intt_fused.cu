/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/ntt_merge_intt_fused.cu
 * ROLE   inverse NTT: pointwise operators, dispatch and public entry points
 *
 * HOW    The Op template supplies the pointwise stage folded into the first
 *        load (multiply by an external spectrum, square, or nothing); the
 *        dispatcher picks the launch configuration and instantiates the
 *        kernels.
 *
 * CHANGELOG
 *   2026-08-11  Kernels moved to ntt/kernels/; this file keeps the gather
 *               ops, the launch-config dispatch and the public entry points.
 * ───────────────────────────────────────────────────────────────────────────── */
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

#include "ntt/ntt_merge_intt_fused.cuh"
#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"

#include "util/const_dispatch.h"
#include "ntt/ntt_merge_kernel_bounds.h"
#include "ntt/shoup_butterfly.cuh"

using namespace gpuntt;

#define GPUNTT_CHECK(expr) \
    do \
    { \
        cudaError_t _e = (expr); \
        if (_e != cudaSuccess) \
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
    Modulus64 modulus;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        return OPERATOR_GPU<Data64>::mult(a[idx], b[idx], modulus);
    }
};

// No pointwise op: the fused INTT then behaves as a plain in-place inverse NTT,
// keeping the Shoup butterflies on that path as well.
struct OpNone
{
    const Data64 *a;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        return a[idx];
    }
};

struct OpSq
{
    const Data64 *a;
    Modulus64 modulus;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        return OPERATOR_GPU<Data64>::mult(a[idx], a[idx], modulus);
    }
};

#include "ntt/kernels/inverse_lowring.cuh"
#include "ntt/kernels/inverse_core.cuh"
#include "ntt/kernels/inverse_core_transposed.cuh"
#include "ntt/kernels/inverse_tail.cuh"


// ─────────────────────────────────────────────────────────────────────────────
// Shared dispatch logic
// ─────────────────────────────────────────────────────────────────────────────

// This project always runs X_N_minus, so only that specialisation is
// instantiated — building both would double an already large template fan-out
// (Op x OIC x SI) for a branch that never executes.
static constexpr bool FUSED_RPC = true;

template <typename Op>
static void dispatch_intt_fused(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size,
    Op op)
{
    if (cfg.ntt_layout != PerPolynomial)
        throw std::runtime_error("[fused_intt] only the PerPolynomial layout is supported");
    if (cfg.reduction_poly != ReductionPolynomial::X_N_minus)
        throw std::runtime_error("[fused_intt] only ReductionPolynomial::X_N_minus is supported");

    auto kernel_params = CreateInverseNTTKernel<Data64>();
    const bool low_ring = (cfg.n_power < 11);
    const bool std_kernel = (cfg.n_power < 25);

    // Both non-low-ring kernels specialise on (outer_iteration_count, shared_index),
    // so their launches share one nested dispatch.
    auto dispatch_pass = [&](const KernelConfig &p, auto &&launch)
    {
        dispatch_const<1, MAX_OUTER_ITERATION_COUNT>(
            p.outer_iteration_count, [&](auto oic)
            { dispatch_const<MIN_SHARED_INDEX, MAX_SHARED_INDEX>(
                  p.shared_index, [&](auto si)
                  { launch(oic, si); }); });
        GPUNTT_CHECK(cudaGetLastError());
    };

    // Every remaining pass is the plain (non-fused) tail kernel.
    auto launch_tail = [&](const KernelConfig &p)
    {
        dispatch_pass(p, [&](auto oic, auto si)
                      { InverseCore_Tail<oic.value, si.value, FUSED_RPC><<<
                            dim3(p.griddim_x, p.griddim_y, batch_size),
                            dim3(p.blockdim_x, p.blockdim_y),
                            p.shared_memory, cfg.stream>>>(
                            device_inout, device_inout,
                            root_of_unity_table, root_shoup_table, modulus,
                            p.logm, p.k,
                            cfg.n_power, cfg.mod_inverse, n_inv_shoup,
                            p.not_last_kernel); });
    };

    if (low_ring)
    {
        // shared_index is unused by this kernel (its t_ starts at 0), so n_power
        // is the only constant it needs.
        auto &p = kernel_params[cfg.n_power][0];
        int grid_x = (batch_size + p.blockdim_y - 1) / p.blockdim_y;
        dispatch_const<1, MAX_LOW_RING_N_POWER>(
            cfg.n_power, [&](auto np)
            { InverseCoreLowRing_Fused<Op, np.value, FUSED_RPC><<<
                  dim3(grid_x, 1, 1),
                  dim3(p.blockdim_x, p.blockdim_y),
                  p.shared_memory, cfg.stream>>>(
                  device_inout,
                  root_of_unity_table, root_shoup_table, modulus,
                  cfg.mod_inverse, n_inv_shoup, batch_size, op); });
        GPUNTT_CHECK(cudaGetLastError());
        return;
    }

    // Pass 0 is the fused kernel: standard block layout below n_power 25,
    // transposed (InverseCore_ equivalent) at or above it.
    {
        auto &p = kernel_params[cfg.n_power][0];
        dispatch_pass(p, [&](auto oic, auto si)
                      {
            if (std_kernel)
                InverseCore_Fused<Op, oic.value, si.value, FUSED_RPC><<<
                    dim3(p.griddim_x, p.griddim_y, batch_size),
                    dim3(p.blockdim_x, p.blockdim_y),
                    p.shared_memory, cfg.stream>>>(
                    device_inout,
                    root_of_unity_table, root_shoup_table, modulus,
                    p.logm, p.k,
                    cfg.n_power, cfg.mod_inverse, n_inv_shoup,
                    p.not_last_kernel, op);
            else
                InverseCore__Fused<Op, oic.value, si.value, FUSED_RPC><<<
                    dim3(p.griddim_x, p.griddim_y, batch_size),
                    dim3(p.blockdim_x, p.blockdim_y),
                    p.shared_memory, cfg.stream>>>(
                    device_inout,
                    root_of_unity_table, root_shoup_table, modulus,
                    p.logm, p.k,
                    cfg.n_power, cfg.mod_inverse, n_inv_shoup,
                    p.not_last_kernel, op); });
    }

    for (int i = 1; i < (int)kernel_params[cfg.n_power].size(); i++)
        launch_tail(kernel_params[cfg.n_power][i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

__host__ void GPU_INTT_Inplace_PreMul(
    Data64 *device_inout,
    const Data64 *b,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpMul op{device_inout, b, modulus};
    dispatch_intt_fused(device_inout, root_of_unity_table, root_shoup_table, n_inv_shoup, modulus, cfg, batch_size, op);
}

__host__ void GPU_INTT_Inplace_PreSq(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpSq op{device_inout, modulus};
    dispatch_intt_fused(device_inout, root_of_unity_table, root_shoup_table, n_inv_shoup, modulus, cfg, batch_size, op);
}

__host__ void GPU_INTT_Inplace_Shoup(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Data64 n_inv_shoup,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpNone op{device_inout};
    dispatch_intt_fused(device_inout, root_of_unity_table, root_shoup_table, n_inv_shoup,
                        modulus, cfg, batch_size, op);
}
