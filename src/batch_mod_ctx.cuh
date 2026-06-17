#pragma once
// batch_mod_ctx.cuh — batched modular arithmetic context (Montgomery / Barrett).
//
// Layout: all arrays [n_batch * stride], batch_i = candidate.
// NTT called a single time with batch=n_batch.

#include "config.h"
#include "ops/mul/multiplier.cuh"
#include "helpers/time_format.h"
#include "perf/perf_node.cuh"
#include "perf/perf_timer.cuh"
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <memory>
#include <gmp.h>
#include <cuda_runtime.h>

// Validation of the chosen reduction algorithm (params.cmake → MOD_REDUCTION_ALG).
#if MOD_REDUCTION_ALG == MOD_RED_BURNIKEL_ZIEGLER
#error "MOD_RED_BURNIKEL_ZIEGLER not yet implemented. Use MOD_RED_MONTGOMERY or MOD_RED_BARRETT in params.cmake."
#elif MOD_REDUCTION_ALG != MOD_RED_MONTGOMERY && MOD_REDUCTION_ALG != MOD_RED_BARRETT
#error "Invalid MOD_REDUCTION_ALG. Values: MOD_RED_MONTGOMERY | MOD_RED_BARRETT | MOD_RED_BURNIKEL_ZIEGLER."
#endif

// Limb headroom of the NTT context. Barrett multiplies operands of up to
// (n_limbs+1) limbs (A1·μ), requiring padded >= 2(k+1)-1; +1 limb guarantees this
// even when 2k is already a power of two. Montgomery uses operands of k limbs.
#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
#define MOD_NTT_EXTRA 1
#else
#define MOD_NTT_EXTRA 0
#endif

// Conversion between a normal integer and the "working form" of the reduction backend
// (defined in reductions/montgomery.cu / reductions/barrett.cu). Montgomery: x·R^{±1} mod N;
// Barrett: plain residue x mod N. res and x are mpz_t; N the modulus; n_limbs the width.
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs);
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs);

// ── Profiling tree indices ───────────────────────────────────────────────
// Used by batch_mod_ctx.cu and the reduction files to navigate perf_cur.
// perf_cur is the context root (mul or sq); its children:
//   child(PERF_PROD) = "product" node   (4 leaves: NTT, pmul, INTT, carry)
//   child(PERF_RED)  = "reduction" node (internal structure varies per algorithm)
//   child(PERF_FIN)  = "finalize" node  (Barrett: 3 leaves; Montgomery: cond_sub leaf)
enum PerfCtxIdx  { PERF_PROD = 0, PERF_RED = 1, PERF_FIN = 2 };
// Children of PERF_PROD:
enum PerfProdIdx { PERF_PROD_NTT = 0, PERF_PROD_PMUL = 1, PERF_PROD_INTT = 2, PERF_PROD_CARRY = 3 };

struct BatchModCtx
{
    int n_limbs, n_batch, padded, n_sum;
    Multiplier ntt; // multiplication backend (compile-time: MUL_ALG)

    // Per-candidate data, [n_batch * n_limbs]
    Data64 *d_N = nullptr;
    Data64 *d_Nprime = nullptr;

    // Pre-computed NTT(N) and NTT(N'), [n_batch * padded] — read-only on the hot path
    Data64 *d_ntt_N = nullptr;
    Data64 *d_ntt_Nprime = nullptr; // MOD_RED_MONTGOMERY only

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    // PER-CANDIDATE Barrett parameter: bar_k[i] = number of "tight" limbs of N_i — the
    // index of the most significant non-zero limb + 1 (b^{k-1} <= N_i < b^{k}).
    // limbs_for_digits() allocates +4 slack limbs and the sparse candidates differ
    // from each other in a few top limbs, so bar_k varies per candidate.
    // The buffer width is uniform: bar_W1 = max_i bar_k[i] + 1.
    int bar_W1 = 0;            // max(bar_k) + 1 (width of A1, μ and q̂)
    int *d_bar_k = nullptr;    // [n_batch] bar_k per candidate (device)
    // μ_i = floor(b^{2·bar_k_i}/N_i) per candidate, already transformed (NTT(μ)).
    Data64 *d_ntt_mu = nullptr; // [n_batch * padded]
    // Barrett reduction scratch (disjoint lifetimes ⇒ reused):
    //   d_bar_w1   [n_batch * bar_W1] — A1 = T>>(k-1), then q̂, then residue r.
    //   d_bar_prod [n_batch * n_sum]  — intermediate product: A1·μ, then q̂·N.
    Data64 *d_bar_w1 = nullptr;
    Data64 *d_bar_prod = nullptr;
#endif

    // Reference values in working form — for checking without GMP.
    // Montgomery: to_mont(·). Barrett: plain residue (1 and N-1). [n_batch*n_limbs]
    Data64 *d_one_res = nullptr; // working form of 1   per candidate
    Data64 *d_Nm1_res = nullptr; // working form of N-1 per candidate

    // Working buffers
    Data64 *d_T = nullptr; // [n_batch * n_sum]
    Data64 *d_m = nullptr; // [n_batch * padded]  (NTT workspace for m, Montgomery only)

    // Tiled subtractor buffers (ops/sub) [n_batch * n_cs_tiles]
    int n_cs_tiles = 0;
    int *d_cs_tile_cmp = nullptr;    // cmp per tile: 1, -1, 0
    int *d_cs_tile_bstate = nullptr; // G/P/K borrow state per tile

    // ── Dynamic profiling (PerfNode tree) ─────────────────────────────────
    // The tree is built in the constructor via build_perf_nodes(). Each sub-group
    // (product, reduction, finalize) is a branch with children created by branch().
    // Children are accessed by index (child(int)) — no struct of fixed pointers.
    // The report simply walks the tree; no field needs to be hardcoded.
    PerfNode perf_root{"TOTAL"};
    PerfTimer timer;

    // Roots of the mul and sq contexts (branches of perf_root).
    // perf_cur points to mul or sq during each public call.
    PerfNode *perf_mul = nullptr;
    PerfNode *perf_sq  = nullptr;
    PerfNode *perf_cur = nullptr;

    // Host phase supplied by the caller (e.g.: setup, table, memcpy). Enters the
    // tree as a synthetic leaf under the "setup / host" group.
    struct HostPhase
    {
        const char *name;
        float ms;
        std::string note; // optional annotation (e.g.: "(17.5 GB/s)")
    };

    // Walks the perf_root graph and prints. app_total_ms fills "others (overhead)";
    // host = host phases grouped under "setup / host". See helpers/mod_perf.cu.
    void print_perf(double app_total_ms = 0.0,
                    const std::vector<HostPhase> &host = {});

    // Enables/disables time collection. When false, TSTART/TSTOP become no-ops.
    bool perf_enabled = false;

    int device_id = 0; // GPU used

    // Constructor from pre-computed limbs.
    // device_id: GPU index (0 by default; use cudaGetDeviceCount to list).
    // N_all: flat vector [n_batch * n_limbs], little-endian 16-bit limbs.
    explicit BatchModCtx(const std::vector<uint64_t> &N_all, int n_limbs_, int n_batch_,
                          int device_id_ = 0);

    // Convenience constructor: accepts the numbers directly as mpz_t.
    // Computes n_limbs automatically from the largest number in the vector.
    explicit BatchModCtx(const std::vector<mpz_t *> &numbers, int device_id_ = 0);
    ~BatchModCtx();

    // x_all (host, n_batch * n_limbs) -> Montgomery form (host)
    void to_residue_batch(const std::vector<uint64_t> &x_all,
                       std::vector<uint64_t> &out_all) const;

    // d_x (GPU, Montgomery form) -> normal values (host)
    void from_residue_batch(const Data64 *d_x, std::vector<uint64_t> &out_all) const;

    // Checks results on the GPU: for each candidate, r_mont == 1_mont or (N-1)_mont?
    // d_passed[t] = 1 if passed, 0 if composite. n_total elements.
    void check_passed(const Data64 *d_r_mont, uint8_t *d_passed, cudaStream_t s = 0) const;

    // d_out = mont_mul(d_A, d_B) for all n_batch candidates
    void modmul_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                        cudaStream_t s = 0);
    // d_out = mont_sq(d_A) for all n_batch candidates
    void modsq_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s = 0);

    // Only NTT(A)*NTT(B) + INTT — no REDC. Measures the pure cost of multiplication.
    void mul_no_redc_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                           cudaStream_t s = 0);
    // Only NTT(A)^2 + INTT — no REDC.
    void sq_no_redc_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s = 0);

    BatchModCtx(const BatchModCtx &) = delete;
    BatchModCtx &operator=(const BatchModCtx &) = delete;

private:
    // Pre-computes and allocates the structures specific to the reduction backend.
    void precompute_reduction(const std::vector<uint64_t> &N_all);
    // Frees what precompute_reduction allocated.
    void free_reduction();
    // Reduces d_T (product in [n_batch*n_sum]) → d_out in working form.
    void reduce_batch(Data64 *d_out, cudaStream_t s);
    // Conditional subtraction mod N (Montgomery only; Barrett finalizes in its own kernel).
    void cond_sub_batch(Data64 *d_x, cudaStream_t s);
    // Synchronizes the last event of the ring and accumulates all pending times.
    void perf_flush(cudaStream_t s);
    // Builds the subtree of one path (mul/sq) under perf_root and returns the root branch.
    PerfNode *build_perf_nodes(const char *ctx_name);
};
