# Multiplication Backends

The big-integer multiplication backend is selected at **compile time** via the
`MUL_ALG` parameter in `params.cmake`. All backends expose the same `Multiplier`
interface (see `src/ops/mul/multiplier.cuh`).

## Overview

| Backend            | Complexity | Max digits (b=16) | External dep         |
| ------------------ | ---------- | ----------------- | -------------------- |
| `MUL_SCHOOLBOOK`   | O(n²)      | unlimited         | —                    |
| `MUL_FFT_CUFFT`    | O(n log n) | ~7.9 × 10⁴        | cuFFT (CUDA Toolkit) |
| `MUL_FFT_GPUFFT`   | O(n log n) | ~7.9 × 10⁴        | GPU-FFT              |
| `MUL_FFNT_GPUFFT`  | O(n log n) | ~7.9 × 10⁴        | GPU-FFT              |
| `MUL_MERGE_GPUNTT` | O(n log n) | ~6.5 × 10⁸        | GPU-NTT              |
| `MUL_4STEP_GPUNTT` | O(n log n) | ~4.0 × 10⁷        | GPU-NTT              |

All backends guarantee correct results. The complex FFT backends (cuFFT,
GPU-FFT, FFNT) use double-precision floating point internally; the constructor
runs a precision guard at startup and aborts with a clear error if the candidate
size would cause a rounding error. The "Max digits" column is therefore the safe
operating range, not a silent precision loss boundary.

> **b** = `LIMB_BITS`. The digit limits assume b=16 (the default). Reducing b
> dramatically extends the FFT backends' safe range — see
> [configuration.md](configuration.md) for the full table.

## Schoolbook — `MUL_SCHOOLBOOK`

Direct O(n²) convolution: each output coefficient is the dot product of two
coefficient vectors, computed by one GPU thread.

- No external library needed.
- No size limit — every input fits.
- Only practical for very small `n_limbs` (< ~64 limbs, i.e. < ~300 digits at
  b=16). Use as a debugging baseline or for quick correctness checks.
- The modular reduction still uses the NTT "merge" backend internally even when
  the product itself uses schoolbook.

## Complex FFT backends

These backends perform polynomial multiplication in the frequency domain using
the classical **complex-valued FFT** (double precision). The FFT is the
conceptual foundation: transform both polynomials, multiply pointwise, inverse
transform. They are typically the fastest option on hardware with strong
complex-arithmetic units (cuFFT in particular is highly optimised).

The precision limit comes from the 52-bit double mantissa. Each coefficient
accumulates rounding error proportional to `n × max_coeff²`; if the error
could exceed 0.5 the integer rounding would be wrong. The constructor enforces:

```
(3.32 × D / b) × log₂(6.64 × D / b)  <  2^(50 − 2b)
```

where `D` = decimal digit count and `b` = `LIMB_BITS`. If the check fails the
program aborts before any wrong result is produced.

**Key insight**: reducing `LIMB_BITS` (e.g. from 16 to 8) makes each
coefficient smaller, dramatically increasing the safe digit range.

### `MUL_FFT_CUFFT`

Wraps NVIDIA's cuFFT library (included in every CUDA Toolkit — no extra
download). Packs two real polynomials into one complex FFT using the standard
"two-for-one real FFT" trick.

- Available everywhere CUDA is installed.
- Safe digit range at b=16: ~79 000 digits. Use b=8 to reach ~1.3 × 10⁹.

### `MUL_FFT_GPUFFT`

Uses the [GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT) library's C2C
(complex-to-complex) transform in "merge" style.

- Requires CMake ≥ 3.26 (GPU-FFT dependency).
- Similar precision range to cuFFT at the same `LIMB_BITS`.
- Additional length constraint: `log₂(n) ≤ 24`.

### `MUL_FFNT_GPUFFT`

Uses GPU-FFT's **negacyclic real FFT** (FFNT). A negacyclic FFT of length n/2
replaces a cyclic FFT of length n, roughly halving the transform work.

- ~2× throughput compared to `MUL_FFT_GPUFFT` for the same polynomial size.
- Same precision and length constraints as `MUL_FFT_GPUFFT`.
- Requires CMake ≥ 3.26.

## Integer FFT backends — NTT

An **NTT (Number Theoretic Transform)** is an FFT evaluated over a finite
field (integers modulo a prime) instead of the complex numbers. All arithmetic
stays in exact integers — there is no mantissa, no rounding, no size-dependent
precision limit. The trade-off is that the NTT prime constrains the transform
length, giving the "Max digits" ceiling in the table above.

### `MUL_MERGE_GPUNTT` — **Default**

Uses the "merge" strategy from the [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)
library. A single transform pass of length up to `2^28` covers the entire
polynomial.

- **Best general choice.** Handles the widest digit range among all backends.
- Requires `log₂(2 × n_limbs) ≤ 28`.
- Memory: one transform-sized buffer per batch slot.

### `MUL_4STEP_GPUNTT` — 4-step radix

Decomposes a large transform into smaller sub-transforms with transpositions
(the "4-step" or "six-step" algorithm).

- Can be faster than merge on very large `n` because the sub-transforms fit
  better in shared memory.
- Stricter size constraint: `log₂(2 × n_limbs) ∈ [12, 24]` → maximum ~4 × 10⁷
  digits at b=16.
- More complex kernel scheduling; uses two GPU transpose passes.

