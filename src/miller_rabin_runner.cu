// miller_rabin_runner.cu — Global witness-sweep Miller-Rabin.
//
// Both functions accept ALL n_total candidates at once.
// For each witness: all alive candidates are tested in sub-batches of MR_BATCH_SIZE,
// then survivors are compacted globally before the next witness.
// This means witness 1 runs across all candidates before witness 2 starts,
// so witness 2 only processes the global survivors of witness 1.

#include "miller_rabin_runner.cuh"
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <algorithm>

#define CU(expr) \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess) \
        throw std::runtime_error(std::string("[CUDA] " #expr ": ")+cudaGetErrorString(_e)); \
    } while(0)

// ── run_one_witness_s1 ────────────────────────────────────────────────────────
// Tests one sub-batch of sub_bsz candidates with one witness.
// Returns a passed[] vector of size sub_bsz.

static std::vector<uint8_t> run_one_witness_s1(
        std::vector<uint64_t>& N_sub,
        std::vector<uint64_t>& exp_sub,
        std::vector<uint64_t>& Nm1_sub,
        int n,
        int sub_bsz,
        uint32_t witness,
        PerfCtrs& perf,
        bool show_progress)
{
    BatchModCtx sub_mont(N_sub, n, sub_bsz);

    WitnessBuffers buf(sub_mont, exp_sub, sub_bsz);

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0)); CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]() {
        float ms = 0;
        CU(cudaEventSynchronize(ev1));
        CU(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms;
    };

    // Build base = witness in Montgomery form
    CU(cudaEventRecord(ev0));
    std::vector<uint64_t> w_all((size_t)sub_bsz * n, 0);
    for (int t = 0; t < sub_bsz; t++) {
        w_all[t*n]   = witness & LIMB_MASK;
        w_all[t*n+1] = (witness >> LIMB_BITS) & LIMB_MASK;
    }
    std::vector<uint64_t> base_mont;
    sub_mont.to_residue_batch(w_all, base_mont);
    CU(cudaEventRecord(ev1));
    perf.setup_ms += elapsed_ms();

    CU(cudaEventRecord(ev0));
    CU(limb_upload(buf.d_base, base_mont.data(), (size_t)sub_bsz * n));
    CU(cudaMemcpy(buf.d_r, buf.d_one, (size_t)sub_bsz * n * sizeof(LimbT), cudaMemcpyDeviceToDevice));
    CU(cudaEventRecord(ev1));
    perf.memcpy_ms    += elapsed_ms();
    perf.memcpy_bytes += (size_t)sub_bsz * n * sizeof(LimbT) * 2;

    window_exp_loop(sub_mont, exp_sub, buf.d_r, buf.d_one, buf.d_base, buf.d_scratch,
                    buf.d_cur_mul, buf.d_exp_dev, sub_bsz, perf, witness, show_progress);

    CU(cudaEventRecord(ev0));
    sub_mont.check_passed(buf.d_r, buf.d_passed);
    std::vector<uint8_t> passed(sub_bsz);
    CU(cudaMemcpy(passed.data(), buf.d_passed, sub_bsz, cudaMemcpyDeviceToHost));
    CU(cudaEventRecord(ev1));
    perf.check_ms += elapsed_ms();

    CU(cudaEventDestroy(ev0)); CU(cudaEventDestroy(ev1));
    return passed;
}

// ── run_one_witness_general ───────────────────────────────────────────────────
// Like run_one_witness_s1 but also performs s-1 extra squarings for s>1.
// Returns a passed[] vector of size sub_bsz.

static std::vector<uint8_t> run_one_witness_general(
        std::vector<uint64_t>& N_sub,
        std::vector<uint64_t>& exp_sub,
        std::vector<uint64_t>& Nm1_sub,
        int n,
        int sub_bsz,
        int s,
        uint32_t witness,
        PerfCtrs& perf,
        bool show_progress)
{
    BatchModCtx sub_mont(N_sub, n, sub_bsz);
    sub_mont.perf_enabled = false;

    WitnessBuffers buf(sub_mont, exp_sub, sub_bsz);

    uint8_t* d_alive;
    CU(cudaMalloc(&d_alive, (size_t)sub_bsz));

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0)); CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]() {
        float ms = 0;
        CU(cudaEventSynchronize(ev1));
        CU(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms;
    };

    CU(cudaEventRecord(ev0));
    std::vector<uint64_t> w_all((size_t)sub_bsz * n, 0);
    for (int t = 0; t < sub_bsz; t++) {
        w_all[t*n]   = witness & LIMB_MASK;
        w_all[t*n+1] = (witness >> LIMB_BITS) & LIMB_MASK;
    }
    std::vector<uint64_t> base_mont;
    sub_mont.to_residue_batch(w_all, base_mont);
    std::vector<uint8_t> alive_h(sub_bsz, 1);
    CU(cudaEventRecord(ev1));
    perf.setup_ms += elapsed_ms();

    CU(cudaEventRecord(ev0));
    CU(limb_upload(buf.d_base, base_mont.data(), (size_t)sub_bsz * n));
    CU(cudaMemcpy(buf.d_r, buf.d_one, (size_t)sub_bsz * n * sizeof(LimbT), cudaMemcpyDeviceToDevice));
    CU(cudaMemcpy(d_alive, alive_h.data(), sub_bsz, cudaMemcpyHostToDevice));
    CU(cudaEventRecord(ev1));
    perf.memcpy_ms    += elapsed_ms();
    perf.memcpy_bytes += (size_t)sub_bsz * n * sizeof(LimbT) * 2 + (size_t)sub_bsz;

    window_exp_loop(sub_mont, exp_sub, buf.d_r, buf.d_one, buf.d_base, buf.d_scratch,
                    buf.d_cur_mul, buf.d_exp_dev, sub_bsz, perf, witness, show_progress);

    // Initial check: r == 1 or r == N-1
    CU(cudaEventRecord(ev0));
    sub_mont.check_passed(buf.d_r, buf.d_passed);
    CU(cudaMemcpy(alive_h.data(), buf.d_passed, sub_bsz, cudaMemcpyDeviceToHost));
    for (int t = 0; t < sub_bsz; t++)
        alive_h[t] = alive_h[t] ? 2 : 1;
    CU(cudaMemcpy(d_alive, alive_h.data(), sub_bsz, cudaMemcpyHostToDevice));
    CU(cudaEventRecord(ev1));
    perf.check_ms += elapsed_ms();

    // Extra squarings: check r^(2^i) == N-1
    for (int sq = 1; sq < s; sq++) {
        sub_mont.modsq_batch(buf.d_r, buf.d_scratch);
        std::swap(buf.d_r, buf.d_scratch);
        check_equals_kernel<<<sub_bsz, MR_THR_CHECK>>>(buf.d_r, sub_mont.d_Nm1_res, d_alive, n, sub_bsz);
    }

    CU(cudaMemcpy(alive_h.data(), d_alive, sub_bsz, cudaMemcpyDeviceToHost));
    CU(cudaEventDestroy(ev0)); CU(cudaEventDestroy(ev1));
    CU(cudaFree(d_alive));

    // Convert: alive_h[t]==2 means passed
    std::vector<uint8_t> passed(sub_bsz);
    for (int t = 0; t < sub_bsz; t++)
        passed[t] = (alive_h[t] == 2) ? 1 : 0;
    return passed;
}

// ── extract_sub_batch ─────────────────────────────────────────────────────────

static void extract_sub_batch(
        const std::vector<uint64_t>& src, int n, int start, int sub_bsz,
        std::vector<uint64_t>& dst)
{
    dst.assign((size_t)sub_bsz * n, 0);
    for (int i = 0; i < sub_bsz; i++)
        std::copy(src.begin() + (start+i)*n, src.begin() + (start+i+1)*n, dst.begin() + i*n);
}

// ── gpu_miller_rabin_s1 ───────────────────────────────────────────────────────

std::vector<bool> gpu_miller_rabin_s1(
        const std::vector<uint64_t>& N_all,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int n_limbs,
        int n_total,
        const std::vector<uint32_t>& witnesses,
        const char* label,
        bool show_report,
        bool show_progress)
{
    int n = n_limbs;
    int batch_size = MR_BATCH_SIZE;

    std::vector<uint64_t> N_cur = N_all;
    std::vector<uint64_t> exp_cur = exp_all;
    std::vector<uint64_t> Nm1_cur = Nm1_all;
    std::vector<int> orig_idx(n_total);
    for (int i = 0; i < n_total; i++) orig_idx[i] = i;

    std::vector<bool> result(n_total, false);
    PerfCtrs perf;
    int n_cur = n_total;

    for (int wi = 0; wi < (int)witnesses.size() && n_cur > 0; wi++) {
        if (show_report) {
            printf("  [%s] Witness %-3u  alive: %d\n", label, witnesses[wi], n_cur);
            fflush(stdout);
        }

        // Test all alive candidates in sub-batches, collect results
        std::vector<uint8_t> passed_all(n_cur, 0);

        for (int start = 0; start < n_cur; start += batch_size) {
            int sub_bsz = std::min(batch_size, n_cur - start);
            std::vector<uint64_t> N_sub, exp_sub, Nm1_sub;
            extract_sub_batch(N_cur,   n, start, sub_bsz, N_sub);
            extract_sub_batch(exp_cur, n, start, sub_bsz, exp_sub);
            extract_sub_batch(Nm1_cur, n, start, sub_bsz, Nm1_sub);

            auto sub_res = run_one_witness_s1(N_sub, exp_sub, Nm1_sub, n, sub_bsz,
                                               witnesses[wi], perf, show_progress);
            for (int i = 0; i < sub_bsz; i++)
                passed_all[start + i] = sub_res[i];
        }

        // Global compact
        std::vector<int> keep;
        for (int t = 0; t < n_cur; t++) {
            if (passed_all[t])
                keep.push_back(t);
            else if (show_report)
                printf("    [%s] entry %d COMPOSITE (witness %u)\n", label, orig_idx[t], witnesses[wi]);
        }

        if ((int)keep.size() < n_cur)
            orig_idx = compact_arrays(keep, n, N_cur, exp_cur, Nm1_cur, orig_idx);
        n_cur = (int)keep.size();
    }


    for (int i = 0; i < n_cur; i++)
        result[orig_idx[i]] = true;
    return result;
}

// ── gpu_miller_rabin ──────────────────────────────────────────────────────────

std::vector<bool> gpu_miller_rabin(
        const std::vector<uint64_t>& N_all,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int s,
        int n_limbs,
        int n_total,
        const std::vector<uint32_t>& witnesses,
        const char* label,
        bool show_report,
        bool show_progress)
{
    if (s == 1)
        return gpu_miller_rabin_s1(N_all, exp_all, Nm1_all, n_limbs, n_total,
                                   witnesses, label, show_report, show_progress);

    int n = n_limbs;
    int batch_size = MR_BATCH_SIZE;

    std::vector<uint64_t> N_cur = N_all;
    std::vector<uint64_t> exp_cur = exp_all;
    std::vector<uint64_t> Nm1_cur = Nm1_all;
    std::vector<int> orig_idx(n_total);
    for (int i = 0; i < n_total; i++) orig_idx[i] = i;

    std::vector<bool> result(n_total, false);
    PerfCtrs perf;
    int n_cur = n_total;

    for (int wi = 0; wi < (int)witnesses.size() && n_cur > 0; wi++) {
        if (show_report) {
            printf("  [%s] Witness %-3u  alive: %d\n", label, witnesses[wi], n_cur);
            fflush(stdout);
        }

        std::vector<uint8_t> passed_all(n_cur, 0);

        for (int start = 0; start < n_cur; start += batch_size) {
            int sub_bsz = std::min(batch_size, n_cur - start);
            std::vector<uint64_t> N_sub, exp_sub, Nm1_sub;
            extract_sub_batch(N_cur,   n, start, sub_bsz, N_sub);
            extract_sub_batch(exp_cur, n, start, sub_bsz, exp_sub);
            extract_sub_batch(Nm1_cur, n, start, sub_bsz, Nm1_sub);

            auto sub_res = run_one_witness_general(N_sub, exp_sub, Nm1_sub, n, sub_bsz,
                                                    s, witnesses[wi], perf, show_progress);
            for (int i = 0; i < sub_bsz; i++)
                passed_all[start + i] = sub_res[i];
        }

        // Global compact
        std::vector<int> keep;
        for (int t = 0; t < n_cur; t++) {
            if (passed_all[t])
                keep.push_back(t);
            else if (show_report)
                printf("    [%s] entry %d COMPOSITE (witness %u)\n", label, orig_idx[t], witnesses[wi]);
        }

        if ((int)keep.size() < n_cur)
            orig_idx = compact_arrays(keep, n, N_cur, exp_cur, Nm1_cur, orig_idx);
        n_cur = (int)keep.size();
    }

    for (int i = 0; i < n_cur; i++)
        result[orig_idx[i]] = true;
    return result;
}
