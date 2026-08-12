/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/ntt_merge_ntt_fused.cu
 * ROLE   forward NTT: gather operators, dispatch and public entry points
 *
 * HOW    The gather Op decides where each coefficient comes from (zero-pad,
 *        per-candidate shifted window, one of two sources, or the buffer
 *        itself); the dispatcher picks the launch configuration for n_power
 *        and instantiates the kernels with it.
 *
 * CHANGELOG
 *   2026-08-11  Kernels moved to ntt/kernels/; this file keeps the gather
 *               ops, the launch-config dispatch and the public entry points.
 * ───────────────────────────────────────────────────────────────────────────── */
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

#include "ntt/ntt_merge_ntt_fused.cuh"
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
            throw std::runtime_error(std::string("[fused_ntt] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Gather ops applied at the two global loads per thread.  `idx` is a linear
// index into the padded transform buffer, so (candidate, coefficient) is
// recovered with a shift/mask against the transform length.
// ─────────────────────────────────────────────────────────────────────────────

struct OpPad
{
    const LimbT *src;
    int n_src;
    int logp; // log2(transform length) == cfg.n_power

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        const location_t j = idx & (((location_t)1 << logp) - 1);
        if (j >= (location_t)n_src)
            return 0ULL;
        return (Data64)limb_ld(src[(idx >> logp) * (location_t)n_src + j]);
    }
};

// Reads the transform buffer itself. Running the fused first pass with this op is
// exactly a plain in-place forward NTT — which is how the in-place entry point gets
// the Shoup butterflies too, instead of falling back to the upstream kernels.
struct OpIdentity
{
    const Data64 *src;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        return src[idx];
    }
};

// Gathers a per-candidate right-shifted window: coefficient j reads
// src[cand*n_src + j + bark[cand] + delta], zero outside [0, n_src) and above
// n_out. This is exactly ops::shift_right_var, folded into the first NTT pass —
// it removes a full read+write of the shifted operand per Barrett step.
struct OpShiftPad
{
    const LimbT *src;
    const int *bark;
    int delta;
    int n_out; // logical width of the shifted operand (W1)
    int n_src;
    int logp;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        const location_t j = idx & (((location_t)1 << logp) - 1);
        if (j >= (location_t)n_out)
            return 0ULL;
        const location_t cand = idx >> logp;
        const long long sidx = (long long)j + bark[cand] + delta;
        if (sidx < 0 || sidx >= (long long)n_src)
            return 0ULL;
        return (Data64)limb_ld(src[cand * (location_t)n_src + (location_t)sidx]);
    }
};

// Candidates [0, n_batch) gather from srcA, the rest from srcB — matches the
// A|B contiguous buffer that ntt_AB transforms in a single launch.
struct OpPad2
{
    const LimbT *srcA;
    const LimbT *srcB;
    int n_src;
    int n_batch;
    int logp;

    __device__ __forceinline__ Data64 operator[](location_t idx) const
    {
        const location_t j = idx & (((location_t)1 << logp) - 1);
        if (j >= (location_t)n_src)
            return 0ULL;
        location_t cand = idx >> logp;
        const LimbT *src = srcA;
        if (cand >= (location_t)n_batch)
        {
            src = srcB;
            cand -= (location_t)n_batch;
        }
        return (Data64)limb_ld(src[cand * (location_t)n_src + j]);
    }
};

#include "ntt/kernels/forward_lowring.cuh"
#include "ntt/kernels/forward_core.cuh"
#include "ntt/kernels/forward_tail.cuh"


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
    const Data64 *root_shoup_table,
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
                    root_of_unity_table, root_shoup_table, modulus,
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
                  device_out, root_of_unity_table, root_shoup_table, modulus,
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
                            root_of_unity_table, root_shoup_table, modulus,
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
    const LimbT *src, int n_src,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpPad op{src, n_src, cfg.n_power};
    dispatch_ntt_fused(device_out, root_of_unity_table, root_shoup_table, modulus, cfg, batch_size, op);
}

__host__ void GPU_NTT_ZeroPadLoad2(
    Data64 *device_out,
    const LimbT *srcA, const LimbT *srcB, int n_src, int n_batch,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpPad2 op{srcA, srcB, n_src, n_batch, cfg.n_power};
    dispatch_ntt_fused(device_out, root_of_unity_table, root_shoup_table, modulus, cfg, batch_size, op);
}

__host__ void GPU_NTT_ShiftPadLoad(
    Data64 *device_out,
    const LimbT *src, const int *bark, int delta, int n_out, int n_src,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpShiftPad op{src, bark, delta, n_out, n_src, cfg.n_power};
    dispatch_ntt_fused(device_out, root_of_unity_table, root_shoup_table, modulus, cfg, batch_size, op);
}

__host__ void GPU_NTT_Inplace_Shoup(
    Data64 *device_inout,
    Root64 *root_of_unity_table,
    const Data64 *root_shoup_table,
    Modulus64 modulus,
    ntt_configuration<Data64> cfg,
    int batch_size)
{
    OpIdentity op{device_inout};
    dispatch_ntt_fused(device_inout, root_of_unity_table, root_shoup_table, modulus,
                       cfg, batch_size, op);
}
