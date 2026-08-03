#pragma once
// cpu_witness.h — single-witness Miller-Rabin check on the CPU, mirroring the
// GPU's gpu_test_witness() granularity exactly: given N-1 = 2^s * d, tests
// one witness (base^d mod N, then up to s-1 squarings) and returns pass/fail.
//
// Exists so the throughput/latency benchmarks compare the same unit of work
// on both sides — "one modular exponentiation witness check" — rather than
// mpz_probab_prime_p's internal (and CPU-only) repetition count, which would
// not correspond to anything the GPU side does per call.

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
