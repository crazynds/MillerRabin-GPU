/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/cpu_witness.h
 * ROLE   one Miller-Rabin witness on the CPU
 *
 * HOW    Given N-1 = 2^s * d, computes base^d mod N and up to s-1
 *        squarings, returning pass/fail — the same granularity as the GPU
 *        gpu_test_witness.
 *
 * NOTE   Exists so both sides are timed on the same unit of work.
 *        mpz_probab_prime_p would instead use its own internal repetition
 *        count, which corresponds to nothing the GPU does per call.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <gmp.h>

struct CpuWitnessCtx {
    mpz_t N, Nm1, d;
    int s = 0;
};

// Precomputes N-1 = 2^s * d once so repeated cpu_test_witness() calls are cheap.
void cpu_witness_ctx_init(CpuWitnessCtx &ctx, const mpz_t N);
void cpu_witness_ctx_clear(CpuWitnessCtx &ctx);

// Runs one Miller-Rabin witness check for the given base. Returns true if N
// looks probably prime with respect to this witness (r==1, r==N-1, or one of
// the s-1 squarings hits N-1).
bool cpu_test_witness(const CpuWitnessCtx &ctx, unsigned long witness);
