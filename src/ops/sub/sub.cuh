// ops/sub/sub.cuh — Tiled big-int subtraction (borrow prefix-scan G/P/K), batched.
//
// Generic subtractor used by ALL reductions (Barrett finalize and Montgomery
// cond_sub). Supports separate strides for a/b/out, per-candidate width of b
// (bk) and unconditional (uncond) or conditional mode (only subtracts if a >= b).
//
// The tile buffers (tile_cmp/tile_bstate) must have n_tiles = ceil(W/MR_SUB_TILE)
// ints per candidate. The borrow_in is resolved INSIDE the apply (fused).
#pragma once

#include "ops/mul/multiplier.cuh" // Data64, LIMB_BITS (via selected backend)
#include <cuda_runtime.h>

namespace ops
{
    // Phase 1: per tile, compares a vs b and writes cmp + borrow state.
    void sub_phase1(const LimbT *a, int sa, const LimbT *b, int sb,
                    const int *bk, int W, int *tile_cmp, int *tile_bstate,
                    int n_batch, cudaStream_t s);

    // Phase 2 (fused with resolve): out = a − b with correct tiled borrow.
    // uncond != 0 ⇒ always subtracts; otherwise only when a >= b (no-op otherwise).
    void sub_apply(LimbT *out, int so, const LimbT *a, int sa, const LimbT *b, int sb,
                   const int *bk, int W, const int *tile_cmp, const int *tile_bstate,
                   int uncond, int n_batch, cudaStream_t s);

    // out[cand*out_limbs + j] = (j < W) ? r[cand*W + j] : 0 — copies low limbs.
    void copy_low(LimbT *out, const LimbT *r, int out_limbs, int W,
                  int n_batch, int thr, cudaStream_t s);

    // number of tiles for a width W (= grid.x of the phases; sizes the buffers).
    int sub_n_tiles(int W);
}
