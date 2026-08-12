/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/mr/witness_buffers.cu
 * ROLE   per-witness device buffers and survivor compaction
 *
 * HOW    WitnessBuffers owns everything one witness sweep needs on the
 *        device and frees it on destruction. compact_arrays rebuilds the
 *        host-side candidate arrays keeping only the survivors, so the next
 *        witness works on a dense batch.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from mr/mr_internals.cu (524 lines).
 * ───────────────────────────────────────────────────────────────────────────── */
#include "mr/mr_internals.cuh"
#include "ops/mul/multiplier.cuh"
#include "util/cuda_check.cuh"

// ── WitnessBuffers constructor/destructor ─────────────────────────────────────

WitnessBuffers::WitnessBuffers(BatchModCtx &mont, const std::vector<uint64_t> &exp_all,
                               int n_total_)
    : n_total(n_total_), n(mont.n_limbs)
{
    size_t count = (size_t)n_total * n;
    size_t tb = count * sizeof(LimbT);
    size_t eb = count * sizeof(Data64);
    CU(cudaMalloc(&d_r, tb));
    CU(cudaMalloc(&d_base, tb));
    CU(cudaMalloc(&d_scratch, tb));
    CU(cudaMalloc(&d_one, tb));
    CU(cudaMalloc(&d_cur_mul, tb));
    CU(cudaMalloc(&d_exp_dev, eb));
    CU(cudaMalloc(&d_passed, (size_t)n_total));
    CU(cudaMemcpy(d_exp_dev, exp_all.data(), eb, cudaMemcpyHostToDevice));

    // 1 in Montgomery form
    std::vector<uint64_t> one_all(count, 0);
    for (int t = 0; t < n_total; t++)
        one_all[t * n] = 1;
    std::vector<uint64_t> one_mont;
    mont.to_residue_batch(one_all, one_mont);
    CU(limb_upload(d_one, one_mont.data(), count));
}

WitnessBuffers::~WitnessBuffers()
{
    cudaFree(d_r);
    cudaFree(d_base);
    cudaFree(d_scratch);
    cudaFree(d_one);
    cudaFree(d_cur_mul);
    cudaFree(d_exp_dev);
    cudaFree(d_passed);
}

// ── compact helpers ───────────────────────────────────────────────────────────

std::vector<int> compact_arrays(
    const std::vector<int> &keep,
    int n,
    std::vector<uint64_t> &N_cur,
    std::vector<uint64_t> &exp_cur,
    std::vector<uint64_t> &Nm1_cur,
    const std::vector<int> &orig_idx)
{
    int new_n = (int)keep.size();
    std::vector<uint64_t> N_new(new_n * n), exp_new(new_n * n), Nm1_new(new_n * n);
    std::vector<int> orig_new(new_n);
    for (int i = 0; i < new_n; i++)
    {
        int src = keep[i];
        std::copy(N_cur.begin() + src * n, N_cur.begin() + (src + 1) * n, N_new.begin() + i * n);
        std::copy(exp_cur.begin() + src * n, exp_cur.begin() + (src + 1) * n, exp_new.begin() + i * n);
        std::copy(Nm1_cur.begin() + src * n, Nm1_cur.begin() + (src + 1) * n, Nm1_new.begin() + i * n);
        orig_new[i] = orig_idx[src];
    }
    N_cur = std::move(N_new);
    exp_cur = std::move(exp_new);
    Nm1_cur = std::move(Nm1_new);
    return orig_new;
}
