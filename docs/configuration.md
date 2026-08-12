# Configuration Reference

All build parameters live in `params.cmake` (copy from `params.cmake.example`).
Edit the file and re-run CMake to apply changes — no source edits required.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Sections follow the same order and grouping as `params.cmake.example`. Defaults
quoted here are the values shipped in that file.

---

## [Number representation]

### `LIMB_BITS` (default: `20`)

Base of the big-integer representation: each limb stores one digit in base
`2^LIMB_BITS`. Valid range **4 – 30**; outside it is a compile-time `#error`.

Raising it packs more bits per limb, so a given number needs fewer limbs and a
shorter transform — but it also **lowers the largest number the build can
test**, sharply. The NTT precision guard requires

```
(padded / 2) · (2^LIMB_BITS − 1)²  <  p        padded = next_pow2(2 · n_limbs)
```

where `p` is the NTT prime (`576460756061519873` ≈ 2⁵⁹ for the merge backend).
The squared term means each extra bit per limb costs roughly a factor of four in
reachable size. A candidate that exceeds the limit is rejected at runtime.

#### Maximum decimal digits — `MUL_MERGE_GPUNTT`

| `LIMB_BITS` | max `padded` | max digits  |
| ----------- | ------------ | ----------- |
| 8           | 2²⁸          | 323 000 000 |
| 12          | 2²⁸          | 484 000 000 |
| 16          | 2²⁸          | 646 000 000 |
| 17          | 2²⁶          | 171 000 000 |
| 18          | 2²⁴          | 45 400 000  |
| 19          | 2²²          | 11 900 000  |
| 20          | 2²⁰          | 3 156 000   |
| 21          | 2¹⁸          | 828 000     |
| 22          | 2¹⁶          | 217 000     |
| 23          | 2¹⁴          | 56 700      |
| 24          | 2¹²          | 14 700      |
| 25          | 2¹⁰          | 3 850       |
| 26          | 2⁸           | 1 000       |

Up to `LIMB_BITS = 16` the binding constraint is the library's `log₂(n) ≤ 28`
transform-length limit rather than precision, which is why those rows share a
`padded` ceiling.

Within the limit, transform length is `next_pow2(2 · n_limbs)`, so a value that
puts `2 · n_limbs` just above a power of two doubles the transform against one
just below it. The binary warns at startup (`[ntt] warning: transform is N%
padding`) and names the `LIMB_BITS` that would reach the smaller transform.

#### Maximum digit sizes per backend (b = `LIMB_BITS`)

| Backend            | b=16       | b=12       | b=8        |
| ------------------ | ---------- | ---------- | ---------- |
| `MUL_SCHOOLBOOK`   | ∞          | ∞          | ∞          |
| `MUL_MERGE_GPUNTT` | ~6.5 × 10⁸ | ~4.8 × 10⁸ | ~3.2 × 10⁸ |
| `MUL_4STEP_GPUNTT` | ~4.0 × 10⁷ | ~3.0 × 10⁷ | ~2.0 × 10⁷ |
| `MUL_FFT_CUFFT`    | ~7.9 × 10⁴ | ~1.0 × 10⁷ | ~1.3 × 10⁹ |
| `MUL_FFT_GPUFFT`   | ~7.9 × 10⁴ | ~1.0 × 10⁷ | ~2.0 × 10⁷ |
| `MUL_FFNT_GPUFFT`  | ~7.9 × 10⁴ | ~1.0 × 10⁷ | ~2.0 × 10⁷ |

The NTT backends are exact. The FFT backends are approximate (double precision)
and abort at runtime if the precision guard predicts a wrong result.

### `MR_LIMB32` (default: `ON`)

Stores limb arrays as `uint32` instead of `uint64`. A normalized limb is
`< 2^LIMB_BITS ≤ 2^32`, so the upper half of every word is zero; dropping it
halves the bytes moved by every limb-side kernel (carry, subtract, shift, copy,
NTT gather) and halves the VRAM of the sliding-window power table. Spectral and
raw-INTT coefficients stay 64-bit.

Requires an integer `MUL_ALG` and `LIMB_BITS ≤ 32`; under Montgomery it also
requires a carry algorithm other than `PREFIX_SCAN`. Silently ignored otherwise.

---

## [Batching and exponentiation]

### `MR_BATCH_SIZE` (default: `256`)

Number of candidates processed in a single GPU launch. All candidates in a batch
share one `BatchModCtx` (GPU buffers, NTT plans, Montgomery/Barrett tables).

VRAM per batch ≈ `MR_BATCH_SIZE × n_limbs × 8 bytes × ~20 buffers`. At 18 000
digits with `LIMB_BITS=16`, `n_limbs ≈ 3 750` → ~20 GB at batch 256.

Candidates with different `n_limbs` cannot share a batch; the driver sub-batches
by limb count automatically.

### `MR_WINDOW_BITS` (default: `8`)

Width of the sliding window used during modular exponentiation. The loop
consumes `MR_WINDOW_BITS` exponent bits at a time against a precomputed table of
`2^MR_WINDOW_BITS` powers.

| Value | Trade-off                                                                    |
| ----- | ---------------------------------------------------------------------------- |
| 4–6   | Fewer precomputed powers, more multiplications per exponent                  |
| 7–8   | Balanced                                                                     |
| 9–10  | Fewer multiplications, but table precomputation dominates for small exponents |

Valid range: **4 – 10**.

---

## [Algorithm selection]

### `MUL_ALG` (default: `MUL_MERGE_GPUNTT`)

Selects the big-integer multiplication backend and decides which external
library CMake fetches. Changing it requires a full rebuild.

| Value              | Algorithm                      | Notes                                                          |
| ------------------ | ------------------------------ | -------------------------------------------------------------- |
| `MUL_MERGE_GPUNTT` | GPU-NTT "merge" (O(n log n))   | The only backend the NTT tuning flags apply to.                |
| `MUL_4STEP_GPUNTT` | GPU-NTT "4-step" radix         | Requires log₂(n) ∈ [12, 24].                                   |
| `MUL_FFT_CUFFT`    | Complex FFT via cuFFT          | Approximate. Suits moderate digit counts with small `LIMB_BITS`. |
| `MUL_FFT_GPUFFT`   | GPU-FFT C2C (Alisah-Ozcan)     | Approximate. Requires CMake ≥ 3.26.                            |
| `MUL_FFNT_GPUFFT`  | GPU-FFT negacyclic real (FFNT) | Approximate, real-valued transform. Requires CMake ≥ 3.26.     |
| `MUL_SCHOOLBOOK`   | Direct O(n²) convolution       | Only tractable for tiny `n_limbs`.                             |

See [backends.md](backends.md) for the detailed comparison.

### `MOD_REDUCTION_ALG` (default: `MOD_RED_BARRETT`)

Modular reduction used for every `modmul` and `modsq`.

| Value                | Algorithm         | Working form                                      |
| -------------------- | ----------------- | ------------------------------------------------- |
| `MOD_RED_BARRETT`    | Barrett reduction | Inputs/outputs are plain residues `x mod N`       |
| `MOD_RED_MONTGOMERY` | Montgomery REDC   | Inputs/outputs are in Montgomery form `x·R mod N` |

Barrett precomputes `μ = floor(b^{2k}/N)` once per batch and has no per-multiply
conversion, at the cost of a wider NTT buffer (`n_limbs + 1` instead of
`n_limbs`). Montgomery converts to and from Montgomery form per batch and reuses
the NTT multiply buffer. `MR_NTT_FUSED_SHIFT` requires Barrett.

### `CARRY_NORM_ALG` (default: `CARRY_ALG_MULTI_TILE`)

After each NTT multiplication the coefficient array must be carried back into
canonical limb form (every limb `< 2^LIMB_BITS`).

| Value                   | Description                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `CARRY_ALG_MULTI_TILE`  | Two launches: phase 1 normalizes `MR_CARRY_TILE` limbs per block in parallel, phase 2 propagates the escaped tile carries.       |
| `CARRY_ALG_SINGLE_TILE` | One block per candidate, all tiles carried in shared memory. No inter-block coordination, lower occupancy.                       |
| `CARRY_ALG_SEQUENTIAL`  | One thread per candidate, fully sequential. Minimal resources.                                                                   |
| `CARRY_ALG_PREFIX_SCAN` | Block-wide prefix scan (Kogge-Stone). Experimental, and incompatible with `MR_LIMB32` under Montgomery.                          |

---

## [MUL_MERGE_GPUNTT NTT tuning]

Every flag here requires `MUL_ALG = MUL_MERGE_GPUNTT` and is silently ignored
otherwise.

### `MR_NTT_FUSED_PAD` (default: `ON`)

Fuses the zero-padded gather into the first forward-NTT kernel, removing the
separate `load_padded_batch` pass — one read plus one write over the whole padded
buffer per transformed operand.

### `MR_NTT_FUSED_PMUL` (default: `ON`)

Fuses the NTT-domain pointwise multiply/square into the first INTT kernel,
removing a global round-trip per multiplication. OFF falls back to a separate
`pmul_batch` launch, which is what `MR_THR_PMUL` sizes.

### `MR_NTT_FUSED_SHIFT` (default: `ON`)

Fuses the Barrett variable right-shift into the forward-NTT gather of the operand
it feeds, removing one read plus one write of that operand per Barrett step — two
per modular multiply. Requires `MR_NTT_FUSED_PAD` and `MOD_RED_BARRETT`.

### `MR_NTT_SHOUP` (default: `ON`)

Shoup/Harvey precomputed-quotient butterflies: each twiddle carries a
precomputed `floor(w·2⁶⁴/p)`, turning the modular multiply into one `__umul64hi`
plus two 64-bit multiplies instead of a full Barrett reduction — three 64-bit
multiplies instead of three 128-bit ones. Costs one extra twiddle table in VRAM,
the same size as the existing one. Requires an NTT prime below 2⁶³.

### `MR_NTT_LAZY` (default: `ON`)

Harvey lazy reduction. Coefficients ride between butterflies in `[0,4p)`
(forward) or `[0,2p)` (inverse) instead of `[0,p)`, so each butterfly costs one
conditional subtract instead of three; the transform returns to `[0,p)` on its
last pass.

Requires `MR_NTT_SHOUP` and an NTT prime with `4p < 2⁶⁴`. The merge prime is
≈2⁵⁹, so `4p ≈ 2⁶¹`. **A prime above 2⁶² breaks this silently** — the margin is
structural but thin.

### `MR_NTT_SPARSE_P` (default: `ON`)

Evaluates the `q*p` step of the Shoup butterfly as shifts and adds, exploiting
that the merge prime is sparse: `p = 2⁵⁹ + 2³² − 2²⁹ + 1`. This adds instructions
on the ALU pipe and removes them from the multiply pipe. It is an exact identity
mod 2⁶⁴, and setup aborts if the backend ever selects a different prime.

Optimizations that were tried here and **rejected** — shared-memory bank
swizzle, additive padding, radix-4 — are documented in
[ntt-bank-conflicts.md](ntt-bank-conflicts.md).

---

## [GPU-NTT library]

Options passed to the external
[GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) library that CMake fetches.

### `GPUNTT_CC89` (default: `ON`)

Kernel tables hand-tuned for **Compute Capability 8.9** (RTX 4090):
shared-memory and grid configurations for `n_power` 27 and 28. Set OFF on any
other device — a 3090 is CC 8.6 — where it otherwise selects configurations
built for different hardware.

### `GPUNTT_NTT_LAYOUT` (default: `PerPolynomial`)

How the batch of polynomials is laid out in the NTT buffers.

| Value            | Layout                                     | Notes                                        |
| ---------------- | ------------------------------------------ | --------------------------------------------- |
| `PerPolynomial`  | One polynomial (candidate) per row         | The only layout the fused kernels support.   |
| `PerCoefficient` | One coefficient index per row (transposed) | Not supported by the fused kernels.          |

---

## [Kernel tiling]

### `MR_CARRY_TILE` (default: `512`)

Tile width — threads per block — for the `SINGLE_TILE` and `MULTI_TILE` carry
kernels. Each thread handles one limb, so this also sets the shared-memory
working set per block. Must be a multiple of 32.

### `MR_CARRY_INTER_THR` (default: `64`)

Threads per block for phase 2 (inter-tile carry) of `CARRY_ALG_MULTI_TILE`. One
thread per candidate walks that candidate's escaped tile carries in order.

### `MR_SUB_TILE` (default: `256`)

Tile width for the conditional subtraction kernel `cond_sub_batch`, which brings
a reduced result below N. `cs_phase1` scans `MR_SUB_TILE` limbs per block to
resolve the borrow chain, `cs_apply` writes the corrected output with the same
tile width. Must be a multiple of 32.

---

## [Kernel thread counts]

Threads per block for individual kernels. All must be multiples of 32.

| Parameter           | Default | Kernel                                                                 |
| ------------------- | ------- | ---------------------------------------------------------------------- |
| `MR_THR_LOAD`       | 512     | `load_padded_batch` — gather `n_limbs` coefficients into the NTT buffer |
| `MR_THR_PMUL`       | 256     | `pmul_batch` / `psq_batch` — unused while `MR_NTT_FUSED_PMUL` is ON     |
| `MR_THR_ADD`        | 256     | `vadd_batch` — element-wise add of raw coefficient buffers              |
| `MR_THR_SELECT_WIN` | 32      | `select_window_kernel` — pick the power-table entry for the window      |
| `MR_THR_CHECK`      | 64      | `check_equals_kernel` — compare r against N−1 or 1                      |
| `MR_THR_COPY`       | 512     | result copy-out                                                        |

---

## [Compiler]

### `MR_MAXRREGCOUNT` (default: `0`)

Maps to `nvcc -maxrregcount N`, limiting registers per thread.

- `0` → no limit; the compiler decides.
- Lower values (64, 48) raise occupancy at the cost of spilling registers to
  local memory.

---

## [Monitoring]

### `MR_PROGRESS_INTERVAL_MS` (default: `2000`)

Minimum milliseconds between progress-bar updates. Only relevant with
`--progress`.

### `MR_ADVANCED_MONITOR` (default: `OFF`)

When ON, prints per-batch carry-normalization statistics (min/max/avg
iterations) after each batch, at the cost of measurable `printf` overhead.
