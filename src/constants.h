// ── Carry normalization ───────────────────────────────────────────────────────
#define CARRY_ALG_SINGLE_TILE 1
#define CARRY_ALG_MULTI_TILE 2
#define CARRY_ALG_SEQUENTIAL 3
#define CARRY_ALG_PREFIX_SCAN 4

// ── Big-integer multiplication (mont_mul / mont_sq) ────────────────────────────
//
// SINGLE algorithm selected by MUL_ALG (params.cmake). Defines both the product
// and, when it is NTT, which backend the Multiplier class resolves (see
// ops/mul/multiplier.cuh).
//
// MUL_SCHOOLBOOK — O(n²) by direct convolution. Only for small n_limbs; baseline.
//                  (The reduction still uses the "merge" NTT internally.)
// MUL_MERGE_GPUNTT  — GPU-NTT "merge", O(n log n). Production (default).
// MUL_4STEP_GPUNTT  — GPU-NTT "4step" (radix; transposes). Requires logn ∈ [12,24].
// MUL_FFT_CUFFT  — Complex FFT (double) via cuFFT. Approximate: rounding error
//                  bounds the size (see the precision guard).
// MUL_FFT_GPUFFT — Complex C2C FFT from the GPU-FFT lib (Alisah-Ozcan), "merge" style.
// MUL_FFNT_GPUFFT   — Real negacyclic FFT (FFNT) from the GPU-FFT lib (~2x; real input).
#define MUL_SCHOOLBOOK 1
#define MUL_MERGE_GPUNTT 2
#define MUL_4STEP_GPUNTT 3
#define MUL_FFT_CUFFT 4
#define MUL_FFT_GPUFFT 5
#define MUL_FFNT_GPUFFT 6

// ── Modular reduction (modmul / modsq) ─────────────────────────────────────────
//
// MOD_RED_MONTGOMERY       — Classic REDC. Working form = Montgomery (x·R mod N).
// MOD_RED_BARRETT          — Barrett reduction. Working form = plain residue
//                            (x mod N). Precomputes μ = floor(b^{2k}/N), reuses NTT.
// MOD_RED_BURNIKEL_ZIEGLER — Burnikel-Ziegler D&C division (not implemented).
#define MOD_RED_MONTGOMERY 1
#define MOD_RED_BARRETT 2
#define MOD_RED_BURNIKEL_ZIEGLER 3
