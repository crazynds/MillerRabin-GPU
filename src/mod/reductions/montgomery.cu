/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/mod/reductions/montgomery.cu
 * ROLE   batched Montgomery reduction (REDC)
 *
 * HOW    Working form is x*R mod N with R = 2^(LIMB_BITS*n_limbs). Computes
 *        m = (T mod R)*N mod R, adds m*N to T, shifts right by R and
 *        conditionally subtracts. Compiled only when MOD_REDUCTION_ALG is
 *        MOD_RED_MONTGOMERY.
 *
 * NOTE   Measured about 1.8x slower than Barrett at 100k digits, so Barrett
 *        is the default.
 *
 * CHANGELOG
 *   2026-08-11  T += m*N is now one fused pass: the addend rides in the
 *               carry kernel load instead of being materialized first, which
 *               also makes MR_LIMB32 viable here.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "mod/batch_mod_ctx.cuh"
#include "util/gmp_helpers.cuh"
#include "util/timers.cuh"
#include "ops/shift/shift.cuh"
#include "ops/sub/sub.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_MONTGOMERY

// Local indices (children of perf_cur->child(PERF_RED)):
namespace
{
    enum MontRedIdx
    {
        MONTR_MUL = 0,
        MONTR_SOMA = 1,
        MONTR_SHIFT = 2
    };
    enum MontMulIdx
    {
        MONTM_NTT_TLOW = 0,
        MONTM_PMUL_NP = 1,
        MONTM_INTT_NP = 2,
        MONTM_CARRY_M = 3,
        MONTM_NTT_M = 4,
        MONTM_PMUL_N = 5,
        MONTM_INTT_N = 6
    };
    enum MontSomaIdx
    {
        MONTS_VADD = 0,
        MONTS_CARRY = 1
    };
}

// ── specific GMP helpers ──────────────────────────────────────────────────────

// N' = R - N^{-1} mod R (REDC correction factor). R = 2^(LIMB_BITS·n).
static void compute_Nprime(uint64_t *Np_out, const uint64_t *N_lims, int n)
{
    mpz_t N, R, Np;
    mpz_init(N);
    mpz_init(R);
    mpz_init(Np);
    limbs_to_mpz(N, N_lims, n);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n));
    if (!mpz_invert(Np, N, R))
        throw std::runtime_error("N has no inverse mod R");
    mpz_sub(Np, R, Np);
    mpz_to_limbs(Np_out, n, Np);
    mpz_clear(N);
    mpz_clear(R);
    mpz_clear(Np);
}

// Montgomery working form: res = x·R mod N.
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs)
{
    mpz_t R;
    mpz_init(R);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
    mpz_mul(res, x, R);
    mpz_mod(res, res, N);
    mpz_clear(R);
}

// Montgomery form output: res = x·R^{-1} mod N.
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs)
{
    mpz_t R, Rinv;
    mpz_init(R);
    mpz_init(Rinv);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
    mpz_invert(Rinv, R, N);
    mpz_mul(res, x, Rinv);
    mpz_mod(res, res, N);
    mpz_clear(R);
    mpz_clear(Rinv);
}

// ── backend setup/teardown ────────────────────────────────────────────────────

void BatchModCtx::precompute_reduction(const std::vector<uint64_t> &N_all)
{
    const size_t nb = (size_t)n_batch * n_limbs * sizeof(LimbT);
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);

    n_cs_tiles = ops::sub_n_tiles(n_limbs);
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));
    CU(cudaMalloc(&d_cs_tile_bin, (size_t)n_batch * n_cs_tiles * sizeof(signed char)));

    CU(cudaMalloc(&d_m, pb));

    CU(cudaMalloc(&d_Nprime, nb));
    CU(cudaMalloc(&d_ntt_Nprime, pb));
    std::vector<uint64_t> Np_all((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        compute_Nprime(Np_all.data() + (size_t)i * n_limbs, N_all.data() + (size_t)i * n_limbs, n_limbs);
    CU(limb_upload(d_Nprime, Np_all.data(), (size_t)n_batch * n_limbs));
    ntt.ntt_A(d_Nprime, n_limbs);
    CU(cudaMemcpy(d_ntt_Nprime, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_m);
    cudaFree(d_Nprime);
    cudaFree(d_ntt_Nprime);
    cudaFree(d_cs_tile_cmp);
    cudaFree(d_cs_tile_bstate);
    cudaFree(d_cs_tile_bin);
}

// ── reduction ─────────────────────────────────────────────────────────────────

// Final conditional subtraction of the REDC: if x >= N, x -= N.
void BatchModCtx::cond_sub_batch(LimbT *d_x, cudaStream_t s)
{
    ops::sub_phase1(d_x, n_limbs, d_N, n_limbs, nullptr, n_limbs,
                    d_cs_tile_cmp, d_cs_tile_bstate, /*need_cmp=*/1, n_batch, s);
    ops::sub_resolve(d_cs_tile_cmp, d_cs_tile_bstate, d_cs_tile_bin, n_limbs,
                     /*uncond=*/0, n_batch, s);
    ops::sub_apply(d_x, n_limbs, d_x, n_limbs, d_N, n_limbs, nullptr, n_limbs,
                   d_cs_tile_bin, n_batch, s);
}

// Montgomery reduction (REDC): given T = A·B in d_T [n_batch * n_sum],
// computes out = T · R^{-1} mod N for each candidate.
void BatchModCtx::reduce_batch(LimbT *d_out, cudaStream_t s)
{

    PerfNode *red = perf_cur->child(PERF_RED);
    PerfNode *mul = red->child(MONTR_MUL);
    PerfNode *soma = red->child(MONTR_SOMA);
    PerfNode *csub = perf_cur->child(PERF_FIN);

    TSTART();
    ops::extract_low(ntt.d_buf_A, d_T, n_limbs, padded, n_sum, n_batch, s);
    ntt.fwd_A(s);
    TSTOP(mul->child(MONTM_NTT_TLOW));


#ifdef MR_NTT_FUSED_PMUL
    TSTART();
    ntt.pmul_ext_and_intt(d_ntt_Nprime, s);
    TSTOP(mul->child(MONTM_INTT_NP));
#else
    TSTART();
    ntt.pmul_ext(d_ntt_Nprime, s);
    TSTOP(mul->child(MONTM_PMUL_NP));
    TSTART();
    ntt.intt_A(s);
    TSTOP(mul->child(MONTM_INTT_NP));
#endif
    TSTART();
    ntt.carry_to_limbs(d_m, n_limbs, s);
    TSTOP(mul->child(MONTM_CARRY_M));

    TSTART();
    ntt.ntt_A(d_m, n_limbs, s);
    TSTOP(mul->child(MONTM_NTT_M));
#ifdef MR_NTT_FUSED_PMUL
    TSTART();
    ntt.pmul_ext_and_intt(d_ntt_N, s);
    TSTOP(mul->child(MONTM_INTT_N));
#else
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(mul->child(MONTM_PMUL_N));
    TSTART();
    ntt.intt_A(s);
    TSTOP(mul->child(MONTM_INTT_N));
#endif

    TSTART();
    ntt.add_raw_buf_and_carry(d_T, n_sum, s);
    TSTOP(soma->child(MONTS_CARRY));

    TSTART();
    ops::shift_right(d_out, d_T, n_limbs, n_limbs, n_sum, n_batch, s);
    TSTOP(red->child(MONTR_SHIFT));

    TSTART();
    cond_sub_batch(d_out, s);
    TSTOP(csub);
}

#endif // MOD_RED_MONTGOMERY
