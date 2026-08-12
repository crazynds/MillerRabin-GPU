/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/single_bench.cu
 * ROLE   latency phase of the self-benchmark
 *
 * HOW    Times one witness check on a single candidate: GPU with a batch of
 *        one, CPU on one thread. Repeats single_iters times and reports the
 *        mean.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "bench/single_bench.h"
#include "bench/cpu_witness.h"
#include "driver/candidate.cuh"
#include "mr/miller_rabin_runner.cuh"
#include <chrono>

using hrc = std::chrono::high_resolution_clock;

LatencyStats run_gpu_single(const mpz_t N, int iters, unsigned long witness)
{
    int digits = (int)mpz_sizeinbase(N, 10) + 4;
    int n_limbs = limbs_for_digits(digits);

    NumberCandidate cand;
    cand.build_from_mpz(N, n_limbs);

    LatencyStats stats;
    for (int i = 0; i < iters; i++) {
        std::vector<uint64_t> N_one(cand.N_lims), d_one(cand.d_lims), Nm1_one(cand.Nm1_lims);
        auto t0 = hrc::now();
        gpu_test_witness(N_one, d_one, Nm1_one, n_limbs, 1, cand.s,
                          (uint32_t)witness, nullptr, nullptr, false);
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(hrc::now() - t0).count() / 1000.0;
        stats.add(ms);
    }
    return stats;
}

LatencyStats run_cpu_single(const mpz_t N, int iters, unsigned long witness)
{
    CpuWitnessCtx ctx;
    cpu_witness_ctx_init(ctx, N);

    LatencyStats stats;
    for (int i = 0; i < iters; i++) {
        auto t0 = hrc::now();
        cpu_test_witness(ctx, witness);
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(hrc::now() - t0).count() / 1000.0;
        stats.add(ms);
    }

    cpu_witness_ctx_clear(ctx);
    return stats;
}
