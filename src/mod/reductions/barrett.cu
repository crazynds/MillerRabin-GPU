/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/mod/reductions/barrett.cu
 * ROLE   batched Barrett reduction
 *
 * HOW    Working form is the plain residue, with mu_i = floor(b^(2k_i)/N_i)
 *        precomputed per candidate. Step 1: A1 = floor(T / b^(k-1)), then
 *        q2 = A1*mu by NTT. Step 2: q_hat = floor(q2 / b^(k+1)), then qn =
 *        q_hat*N by NTT. Step 3: out = T - qn, then up to two conditional
 *        subtractions of N and a copy of the low n_limbs.
 *
 * NOTE   Step 2 normalizes only the low W1 limbs of qn — the high ones
 *        cancel in T - qn — so the carry writes with stride W1 and the
 *        subtraction must read qn with that same stride, not n_sum, or the
 *        per-candidate offsets diverge.
 *
 * NOTE   The two reduction multiplies cost twice the product itself, which
 *        makes this the largest single block of GPU time in a run. Compiled
 *        only when MOD_REDUCTION_ALG is MOD_RED_BARRETT.
 *
 * CHANGELOG
 *   2026-08-11  The variable right shift feeding each reduction multiply is
 *               now fused into the forward-NTT gather (MR_NTT_FUSED_SHIFT),
 *               removing one read and one write of the shifted operand per
 *               step.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "mod/batch_mod_ctx.cuh"
#include "util/gmp_helpers.cuh"
#include "util/timers.cuh"
#include "ops/shift/shift.cuh"
#include "ops/sub/sub.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT

// Local indices (children of perf_cur->child(PERF_RED)):
namespace
{
    enum BarIdx
    {
        BAR_SHIFT = 0,
        BAR_Q2 = 1,
        BAR_QN = 2
    };
    enum QIdx
    {
        Q_NTT = 0,
        Q_PMUL = 1,
        Q_INTT = 2,
        Q_CARRY = 3
    };
    enum FinIdx
    {
        FIN_SUB = 0,
        FIN_CONDSUB = 1,
        FIN_COPY = 2
    };
}

// ── specific GMP helpers ──────────────────────────────────────────────────────

// μ = floor(b^{2k}/N), base b = 2^LIMB_BITS. Writes k+1 limbs (little-endian).
static void compute_barrett_mu(uint64_t *mu_out, const uint64_t *N_lims, int k)
{
    mpz_t N, B2k, mu;
    mpz_init(N);
    mpz_init(B2k);
    mpz_init(mu);
    limbs_to_mpz(N, N_lims, k);
    if (mpz_sgn(N) == 0)
        throw std::runtime_error("Barrett: N == 0");
    mpz_ui_pow_ui(B2k, 2, (unsigned long)(LIMB_BITS * 2 * k));
    mpz_tdiv_q(mu, B2k, N);
    mpz_to_limbs(mu_out, k + 1, mu);
    mpz_clear(N);
    mpz_clear(B2k);
    mpz_clear(mu);
}

// Barrett working form = plain residue: res = x mod N (forward and backward identical).
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int)
{
    mpz_mod(res, x, N);
}
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int)
{
    mpz_mod(res, x, N);
}

// ── backend setup/teardown ────────────────────────────────────────────────────

void BatchModCtx::precompute_reduction(const std::vector<uint64_t> &N_all)
{
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);
    const size_t sb = (size_t)n_batch * n_sum * sizeof(LimbT);

    std::vector<int> bar_k_all(n_batch, 0);
    int kmax = 0;
    for (int i = 0; i < n_batch; i++)
    {
        const uint64_t *Ni = N_all.data() + (size_t)i * n_limbs;
        int tight = 0;
        for (int j = n_limbs - 1; j >= 0; j--)
            if (Ni[j] != 0)
            {
                tight = j + 1;
                break;
            }
        if (tight < 2)
            throw std::runtime_error("Barrett: N too small (tight < 2 limbs).");
        bar_k_all[i] = tight;
        if (tight > kmax)
            kmax = tight;
    }
    bar_W1 = kmax + 1;
    CU(cudaMalloc(&d_bar_k, (size_t)n_batch * sizeof(int)));
    CU(cudaMemcpy(d_bar_k, bar_k_all.data(), (size_t)n_batch * sizeof(int), cudaMemcpyHostToDevice));

    const size_t w1b = (size_t)n_batch * bar_W1 * sizeof(LimbT);
    CU(cudaMalloc(&d_ntt_mu, pb));
    CU(cudaMalloc(&d_bar_w1, w1b));
    CU(cudaMalloc(&d_bar_prod, sb));

    std::vector<uint64_t> mu_all((size_t)n_batch * bar_W1, 0);
    for (int i = 0; i < n_batch; i++)
        compute_barrett_mu(mu_all.data() + (size_t)i * bar_W1,
                           N_all.data() + (size_t)i * n_limbs, bar_k_all[i]);
    LimbT *d_mu_tmp = nullptr;
    CU(cudaMalloc(&d_mu_tmp, w1b));
    CU(limb_upload(d_mu_tmp, mu_all.data(), (size_t)n_batch * bar_W1));
    ntt.ntt_A(d_mu_tmp, bar_W1);
    CU(cudaMemcpy(d_ntt_mu, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
    CU(cudaFree(d_mu_tmp));

    n_cs_tiles = ops::sub_n_tiles(bar_W1);
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));
    CU(cudaMalloc(&d_cs_tile_bin, (size_t)n_batch * n_cs_tiles * sizeof(signed char)));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_bar_k);
    cudaFree(d_ntt_mu);
    cudaFree(d_bar_w1);
    cudaFree(d_bar_prod);
    cudaFree(d_cs_tile_cmp);
    cudaFree(d_cs_tile_bstate);
    cudaFree(d_cs_tile_bin);
}

// cond_sub_batch is not used in Barrett (finalize does the subtraction), but the
// function is declared in the header; we provide an empty definition to satisfy the linker.
void BatchModCtx::cond_sub_batch(LimbT *, cudaStream_t) {}

// ── reduction ─────────────────────────────────────────────────────────────────

// Barrett reduction: out = T mod N, with T = A·B in d_T [n_batch*n_sum].
//   q̂ = floor( floor(T/b^{k-1})·μ / b^{k+1} )   (q̂ ∈ {q, q-1, q-2}), k = bar_k[i].
//   out = T − q̂·N, with up to 2 final subtractions of N.
void BatchModCtx::reduce_batch(LimbT *d_out, cudaStream_t s)
{
    const int W1 = bar_W1;

    PerfNode *red = perf_cur->child(PERF_RED);
    PerfNode *q2 = red->child(BAR_Q2);
    PerfNode *qn = red->child(BAR_QN);
    PerfNode *fin = perf_cur->child(PERF_FIN);

#ifdef MR_NTT_FUSED_SHIFT
    TSTART();
    ntt.ntt_A_shifted(d_T, d_bar_k, -1, W1, n_sum, s);
    TSTOP(q2->child(Q_NTT));
#else
    TSTART();
    ops::shift_right_var(d_bar_w1, d_T, d_bar_k, -1, W1, n_sum, n_batch, s);
    TSTOP(red->child(BAR_SHIFT));
    TSTART();
    ntt.ntt_A(d_bar_w1, W1, s);
    TSTOP(q2->child(Q_NTT));
#endif

#ifdef MR_NTT_FUSED_PMUL
    TSTART();
    ntt.pmul_ext_and_intt(d_ntt_mu, s);
    TSTOP(q2->child(Q_INTT));
#else
    TSTART();
    ntt.pmul_ext(d_ntt_mu, s);
    TSTOP(q2->child(Q_PMUL));
    TSTART();
    ntt.intt_A(s);
    TSTOP(q2->child(Q_INTT));
#endif
    TSTART();
    ntt.carry_to_limbs(d_bar_prod, n_sum, s);
    TSTOP(q2->child(Q_CARRY));

#ifdef MR_NTT_FUSED_SHIFT
    TSTART();
    ntt.ntt_A_shifted(d_bar_prod, d_bar_k, +1, W1, n_sum, s);
    TSTOP(qn->child(Q_NTT));
#else
    TSTART();
    ops::shift_right_var(d_bar_w1, d_bar_prod, d_bar_k, +1, W1, n_sum, n_batch, s);
    TSTOP(red->child(BAR_SHIFT));
    TSTART();
    ntt.ntt_A(d_bar_w1, W1, s);
    TSTOP(qn->child(Q_NTT));
#endif

#ifdef MR_NTT_FUSED_PMUL
    TSTART();
    ntt.pmul_ext_and_intt(d_ntt_N, s);
    TSTOP(qn->child(Q_INTT));
#else
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(qn->child(Q_PMUL));
    TSTART();
    ntt.intt_A(s);
    TSTOP(qn->child(Q_INTT));
#endif
    TSTART();
    ntt.carry_to_limbs(d_bar_prod, W1, s);
    TSTOP(qn->child(Q_CARRY));

    TSTART();
    ops::sub_phase1(d_T, n_sum, d_bar_prod, W1, nullptr, W1,
                    d_cs_tile_cmp, d_cs_tile_bstate, /*need_cmp=*/0, n_batch, s);
    ops::sub_resolve(d_cs_tile_cmp, d_cs_tile_bstate, d_cs_tile_bin, W1,
                     /*uncond=*/1, n_batch, s);
    ops::sub_apply(d_bar_w1, W1, d_T, n_sum, d_bar_prod, W1, nullptr, W1,
                   d_cs_tile_bin, n_batch, s);
    TSTOP(fin->child(FIN_SUB));

    TSTART();
    for (int it = 0; it < 2; it++)
    {
        ops::sub_phase1(d_bar_w1, W1, d_N, n_limbs, d_bar_k, W1,
                        d_cs_tile_cmp, d_cs_tile_bstate, /*need_cmp=*/1, n_batch, s);
        ops::sub_resolve(d_cs_tile_cmp, d_cs_tile_bstate, d_cs_tile_bin, W1,
                         /*uncond=*/0, n_batch, s);
        ops::sub_apply(d_bar_w1, W1, d_bar_w1, W1, d_N, n_limbs, d_bar_k, W1,
                       d_cs_tile_bin, n_batch, s);
    }
    TSTOP(fin->child(FIN_CONDSUB));

    TSTART();
    ops::copy_low(d_out, d_bar_w1, n_limbs, W1, n_batch, s);
    TSTOP(fin->child(FIN_COPY));
}

#endif // MOD_RED_BARRETT
