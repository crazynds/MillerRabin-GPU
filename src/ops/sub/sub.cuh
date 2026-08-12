/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/sub/sub.cuh
 * ROLE   batched tiled big-integer subtraction
 *
 * HOW    The subtractor every reduction uses (Barrett finalize, Montgomery
 *        cond_sub). Supports separate strides for a, b and out, a per-
 *        candidate width for b, and either unconditional or conditional
 *        mode, the latter subtracting only when a >= b.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/mul/multiplier.cuh"
#include <cuda_runtime.h>

namespace ops
{
    // Phase 1: per tile, the borrow state and — only if need_cmp — the a vs b
    // comparison. Pass need_cmp = 0 for unconditional subtractions.
    void sub_phase1(const LimbT *a, int sa, const LimbT *b, int sb,
                    const int *bk, int W, int *tile_cmp, int *tile_bstate,
                    int need_cmp, int n_batch, cudaStream_t s);

    // Resolve: per-tile borrow_in into tile_bin (-1 ⇒ a < b, subtraction skipped).
    // uncond != 0 ⇒ always subtracts and tile_cmp is not read.
    void sub_resolve(const int *tile_cmp, const int *tile_bstate, signed char *tile_bin,
                     int W, int uncond, int n_batch, cudaStream_t s);

    // Apply: out = a − b using the resolved per-tile borrow_in. Where tile_bin is
    // -1 the block is a no-op, so in-place callers keep `a` untouched.
    void sub_apply(LimbT *out, int so, const LimbT *a, int sa, const LimbT *b, int sb,
                   const int *bk, int W, const signed char *tile_bin,
                   int n_batch, cudaStream_t s);

    // out[cand*out_limbs + j] = (j < W) ? r[cand*W + j] : 0 — copies low limbs.
    void copy_low(LimbT *out, const LimbT *r, int out_limbs, int W,
                  int n_batch, cudaStream_t s);

    // number of tiles for a width W (= grid.x of the phases; sizes the buffers).
    int sub_n_tiles(int W);
}
