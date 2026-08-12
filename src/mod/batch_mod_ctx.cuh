/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/mod/batch_mod_ctx.cuh
 * ROLE   batched modular arithmetic context
 *
 * HOW    Owns every device buffer a batch needs and the precomputed
 *        reduction tables. All arrays are [n_batch * stride] with the batch
 *        index outermost, so one NTT call transforms the whole batch.
 *
 * NOTE   Buffers, by stride: n_limbs holds N and the working-form
 *        references for 1 and N-1; padded holds the precomputed spectra (of
 *        N, and of N-prime or mu depending on the reduction), read-only on
 *        the hot path; n_sum holds the product T. N_host mirrors d_N on the
 *        host so the residue conversions never pull it back over PCIe,
 *        which was 34 MB per call at 100k digits.
 *
 * NOTE   Barrett adds bar_k[i], the tight limb count of N_i: the index of
 *        its top non-zero limb plus one, so b^(k-1) <= N_i < b^k. It varies
 *        per candidate because limbs_for_digits over-allocates by four and
 *        the sparse candidates differ only in a few top limbs. The scratch
 *        buffers d_bar_w1 and d_bar_prod are shared across steps whose
 *        lifetimes do not overlap.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "config.h"
#include "ops/mul/multiplier.cuh"
#include "util/time_format.h"
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
    Multiplier ntt;

    LimbT *d_N = nullptr;
    LimbT *d_Nprime = nullptr;

    std::vector<uint64_t> N_host;

    Data64 *d_ntt_N = nullptr;
    Data64 *d_ntt_Nprime = nullptr;

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    int bar_W1 = 0;
    int *d_bar_k = nullptr;
    Data64 *d_ntt_mu = nullptr;
    LimbT *d_bar_w1 = nullptr;
    LimbT *d_bar_prod = nullptr;
#endif

    LimbT *d_one_res = nullptr;
    LimbT *d_Nm1_res = nullptr;

    LimbT *d_T = nullptr;
    LimbT *d_m = nullptr;

    int n_cs_tiles = 0;
    int *d_cs_tile_cmp = nullptr;
    int *d_cs_tile_bstate = nullptr;
    signed char *d_cs_tile_bin = nullptr;

    PerfNode perf_root{"TOTAL"};
    PerfTimer timer;

    PerfNode *perf_mul = nullptr;
    PerfNode *perf_sq  = nullptr;
    PerfNode *perf_cur = nullptr;

    // Host phase supplied by the caller (e.g.: setup, table, memcpy). Enters the
    // tree as a synthetic leaf under the "setup / host" group.
    struct HostPhase
    {
        const char *name;
        float ms;
        std::string note;
    };

    // Walks the perf_root graph and prints. app_total_ms fills "others (overhead)";
    // host = host phases grouped under "setup / host". See helpers/mod_perf.cu.
    void print_perf(double app_total_ms = 0.0,
                    const std::vector<HostPhase> &host = {});

    bool perf_enabled = false;

    int device_id = 0;

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
    void from_residue_batch(const LimbT *d_x, std::vector<uint64_t> &out_all) const;

    // Checks results on the GPU: for each candidate, r_mont == 1_mont or (N-1)_mont?
    // d_passed[t] = 1 if passed, 0 if composite. n_total elements.
    void check_passed(const LimbT *d_r_mont, uint8_t *d_passed, cudaStream_t s = 0) const;

    // d_out = mont_mul(d_A, d_B) for all n_batch candidates
    void modmul_batch(const LimbT *d_A, const LimbT *d_B, LimbT *d_out,
                        cudaStream_t s = 0);
    // d_out = mont_sq(d_A) for all n_batch candidates
    void modsq_batch(const LimbT *d_A, LimbT *d_out, cudaStream_t s = 0);

    // Only NTT(A)*NTT(B) + INTT — no REDC. Measures the pure cost of multiplication.
    void mul_no_redc_batch(const LimbT *d_A, const LimbT *d_B, LimbT *d_out,
                           cudaStream_t s = 0);
    // Only NTT(A)^2 + INTT — no REDC.
    void sq_no_redc_batch(const LimbT *d_A, LimbT *d_out, cudaStream_t s = 0);

    BatchModCtx(const BatchModCtx &) = delete;
    BatchModCtx &operator=(const BatchModCtx &) = delete;

private:
    // Pre-computes and allocates the structures specific to the reduction backend.
    void precompute_reduction(const std::vector<uint64_t> &N_all);
    // Frees what precompute_reduction allocated.
    void free_reduction();
    // Reduces d_T (product in [n_batch*n_sum]) → d_out in working form.
    void reduce_batch(LimbT *d_out, cudaStream_t s);
    // Conditional subtraction mod N (Montgomery only; Barrett finalizes in its own kernel).
    void cond_sub_batch(LimbT *d_x, cudaStream_t s);
    // Synchronizes the last event of the ring and accumulates all pending times.
    void perf_flush(cudaStream_t s);
    PerfNode *build_perf_nodes(const char *ctx_name);
};
