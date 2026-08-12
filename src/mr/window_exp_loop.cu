/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/mr/window_exp_loop.cu
 * ROLE   sliding-window modular exponentiation — the hot loop
 *
 * HOW    Precomputes base^w for w < 2^WINDOW_BITS, then walks the exponent
 *        from the top: WINDOW_BITS squarings per window, plus one multiply
 *        by the selected table entry when the window is non-zero. Every
 *        modular operation in a run comes from here.
 *
 * NOTE   The table costs 2^WINDOW_BITS x n_batch x n_limbs limbs of VRAM —
 *        at 100k digits and batch 256 that is the single largest allocation
 *        in the process.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from mr/mr_internals.cu (524 lines).
 * ───────────────────────────────────────────────────────────────────────────── */
#include "mr/mr_internals.cuh"
#include "ops/mul/multiplier.cuh"
#include "util/cuda_check.cuh"
#include "util/time_format.h"
#include <chrono>
#include <cstdio>

using hrc = std::chrono::high_resolution_clock;

// ── Sliding-window exponentiation loop ───────────────────────────────

void window_exp_loop(
    BatchModCtx &mont,
    const std::vector<uint64_t> &exp_all,
    LimbT *&d_r,
    LimbT *d_one_res_h,
    LimbT *d_base,
    LimbT *&d_scratch,
    LimbT *d_cur_mul,
    Data64 *d_exp_dev,
    int n_total,
    PerfCtrs &perf,
    uint32_t witness,
    bool show_progress,
    bool collect_perf)
{
    int n = mont.n_limbs;
    size_t total_bytes = (size_t)n_total * n * sizeof(LimbT);

    LimbT *d_table;
    CU(cudaMalloc(&d_table, (size_t)WINDOW_SIZE * total_bytes));

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0));
    CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]()
    { float ms=0; CU(cudaEventSynchronize(ev1)); CU(cudaEventElapsedTime(&ms,ev0,ev1)); return ms; };

    CU(cudaEventRecord(ev0));
    CU(cudaMemcpy(d_table, d_one_res_h, total_bytes, cudaMemcpyDeviceToDevice));
    CU(cudaMemcpy(d_table + (size_t)1 * n_total * n, d_base, total_bytes, cudaMemcpyDeviceToDevice));
    for (int w = 2; w < WINDOW_SIZE; w++)
        mont.modmul_batch(d_table + (size_t)(w - 1) * n_total * n, d_base, d_table + (size_t)w * n_total * n);
    CU(cudaEventRecord(ev1));
    perf.table_ms += elapsed_ms();

    int msb = n * LIMB_BITS - 1;
    while (msb > 0)
    {
        int li = msb / LIMB_BITS, bit = msb % LIMB_BITS;
        bool any = false;
        for (int t = 0; t < n_total && !any; t++)
            if ((exp_all[t * n + li] >> bit) & 1)
                any = true;
        if (any)
            break;
        msb--;
    }

    int n_windows = msb / WINDOW_BITS + 1;
    int start_win = n_windows * WINDOW_BITS - 1;

    std::vector<bool> any_nonzero(n_windows, false);
    for (int wi = 0; wi < n_windows; wi++)
    {
        int i = start_win - wi * WINDOW_BITS;
        for (int t = 0; t < n_total && !any_nonzero[wi]; t++)
            for (int b = 0; b < WINDOW_BITS && !any_nonzero[wi]; b++)
            {
                int bp = i - b;
                if (bp >= 0 && bp <= msb)
                {
                    if ((exp_all[t * n + bp / LIMB_BITS] >> (bp % LIMB_BITS)) & 1)
                        any_nonzero[wi] = true;
                }
            }
    }

    const int thr = MR_THR_SELECT_WIN;
    dim3 grid_sel((unsigned)(n + thr - 1) / thr, (unsigned)n_total);

    auto t_start = hrc::now();
    auto t_last_print = t_start;
    int last_print_bits = 0;

    for (int win = 0; win < n_windows; win++)
    {
        int i = start_win - win * WINDOW_BITS;

        hrc::time_point t_sq0;
        if (collect_perf) t_sq0 = hrc::now();
        for (int sq = 0; sq < WINDOW_BITS; sq++)
        {
            mont.modsq_batch(d_r, d_scratch);
            std::swap(d_r, d_scratch);
        }
        if (collect_perf)
        {
            perf.sq_ms += std::chrono::duration<float, std::milli>(hrc::now() - t_sq0).count();
            perf.sq_calls += WINDOW_BITS;
        }

        if (any_nonzero[win])
        {
            hrc::time_point t_mul0;
            if (collect_perf) t_mul0 = hrc::now();
            select_window_kernel<<<grid_sel, thr>>>(d_cur_mul, d_table, d_exp_dev, i, WINDOW_BITS, n, n_total);
            mont.modmul_batch(d_r, d_cur_mul, d_scratch);
            std::swap(d_r, d_scratch);
            if (collect_perf)
            {
                perf.mul_ms += std::chrono::duration<float, std::milli>(hrc::now() - t_mul0).count();
                perf.mul_calls++;
            }
        }

        if (show_progress)
        {
            auto now = hrc::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - t_last_print).count() >= MR_PROGRESS_INTERVAL_MS || win == n_windows - 1)
            {
                int done_bits = (win + 1) * WINDOW_BITS;
                int total_bits = n_windows * WINDOW_BITS;
                double ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
                double dms = std::chrono::duration<double, std::milli>(now - t_last_print).count();
                int dbits = done_bits - last_print_bits;
                printf("\r    bit %d/%d  %3d%%  %s  %s/iter   ",
                       done_bits, total_bits, done_bits * 100 / total_bits,
                       fmt_time_ms(ms).c_str(),
                       fmt_time_ms(dbits > 0 ? dms / dbits : 0.0).c_str());
                fflush(stdout);
                t_last_print = now;
                last_print_bits = done_bits;
            }
        }
    }
    if (show_progress)
        printf("\n");

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_table);
}
