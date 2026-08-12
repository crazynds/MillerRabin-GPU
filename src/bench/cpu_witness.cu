/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/cpu_witness.cu
 * ROLE   one Miller-Rabin witness on the CPU
 *
 * HOW    mpz_powm for base^d mod N, then up to s-1 squarings looking for
 *        N-1. Deliberately the same unit of work as one GPU witness call —
 *        see cpu_witness.h.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "bench/cpu_witness.h"

void cpu_witness_ctx_init(CpuWitnessCtx &ctx, const mpz_t N)
{
    mpz_inits(ctx.N, ctx.Nm1, ctx.d, nullptr);
    mpz_set(ctx.N, N);
    mpz_sub_ui(ctx.Nm1, N, 1);

    ctx.s = 0;
    mpz_set(ctx.d, ctx.Nm1);
    while (mpz_even_p(ctx.d)) {
        mpz_tdiv_q_2exp(ctx.d, ctx.d, 1);
        ctx.s++;
    }
}

void cpu_witness_ctx_clear(CpuWitnessCtx &ctx)
{
    mpz_clears(ctx.N, ctx.Nm1, ctx.d, nullptr);
}

bool cpu_test_witness(const CpuWitnessCtx &ctx, unsigned long witness)
{
    mpz_t base, r;
    mpz_init_set_ui(base, witness);
    mpz_init(r);

    mpz_powm(r, base, ctx.d, ctx.N);

    bool passed = (mpz_cmp_ui(r, 1) == 0) || (mpz_cmp(r, ctx.Nm1) == 0);
    for (int i = 1; i < ctx.s && !passed; i++) {
        mpz_powm_ui(r, r, 2, ctx.N);
        if (mpz_cmp(r, ctx.Nm1) == 0)
            passed = true;
    }

    mpz_clear(base);
    mpz_clear(r);
    return passed;
}
