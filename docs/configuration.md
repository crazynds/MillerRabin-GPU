# Configuration Reference

All build parameters live in `params.cmake` (copy from `params.cmake.example`
if you don't have one). Edit the file and re-run CMake to apply changes — no
source edits required.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## Algorithm parameters

### `MR_WINDOW_BITS` (default: `8`)

Width of the sliding window used during modular exponentiation.

The exponentiation loop processes `MR_WINDOW_BITS` bits of the exponent at a
time, looking up a pre-computed power table of size `2^MR_WINDOW_BITS`. A wider
window reduces the number of multiplications but increases the table
pre-computation cost and VRAM usage.

| Value | Trade-off |
|-------|-----------|
| 4–6 | Fewer pre-computed powers, more multiplications per exponent |
| 7–8 | Good balance for most GPU occupancies **(recommended)** |
| 9–10 | Fewer multiplications but table pre-computation dominates for small exponents |

Valid range: **4 – 10**.

---

### `MR_BATCH_SIZE` (default: `256`)

Number of candidates processed in a single GPU launch.

All candidates in a batch share one `BatchModCtx` (GPU buffers, NTT plans,
Montgomery/Barrett pre-computations). Larger batches keep the GPU more occupied
but require more VRAM:

- VRAM per batch ≈ `MR_BATCH_SIZE × n_limbs × 8 bytes × ~20 buffers`
- For 18 000-digit numbers with `LIMB_BITS=16`: n_limbs ≈ 3 750 → ~20 GB for
  batch=256. Reduce if you get OOM errors.

---

### `LIMB_BITS` (default: `16`)

Base of the big-integer representation. Each limb stores one "digit" in base
`2^LIMB_BITS`.

- **Smaller** (e.g. 8) → more limbs → more NTT work, but much more precision
  headroom for the FFT backends (double-precision rounding is safer with smaller
  coefficients).
- **Larger** (e.g. 20) → fewer limbs → faster NTT, but FFT backends overflow
  sooner and may produce wrong results (detected by the runtime precision guard).

Valid range: **4 – 30**. A value outside this range is a compile-time `#error`.

#### Maximum digit sizes per backend (b = LIMB_BITS)

| Backend | b=16 | b=12 | b=8 |
|---------|------|------|-----|
| `MUL_SCHOOLBOOK` | ∞ | ∞ | ∞ |
| `MUL_MERGE_GPUNTT` | ~6.5 × 10⁸ | ~4.8 × 10⁸ | ~3.2 × 10⁸ |
| `MUL_4STEP_GPUNTT` | ~4.0 × 10⁷ | ~3.0 × 10⁷ | ~2.0 × 10⁷ |
| `MUL_FFT_CUFFT` | ~7.9 × 10⁴ | ~1.0 × 10⁷ | ~1.3 × 10⁹ |
| `MUL_FFT_GPUFFT` | ~7.9 × 10⁴ | ~1.0 × 10⁷ | ~2.0 × 10⁷ |
| `MUL_FFNT_GPUFFT` | ~7.9 × 10⁴ | ~1.0 × 10⁷ | ~2.0 × 10⁷ |

> The NTT backends are exact (integer arithmetic). The FFT backends are
> approximate (double-precision floating point) and abort at runtime if the
> precision guard predicts a wrong result.

---

### `MUL_ALG` (default: `MUL_MERGE_GPUNTT`)

Compile-time selection of the big-integer multiplication algorithm.
Changing this requires a full rebuild.

| Value | Algorithm | Notes |
|-------|-----------|-------|
| `MUL_MERGE_GPUNTT` | GPU-NTT "merge" (O(n log n)) | **Default. Best general choice.** |
| `MUL_4STEP_GPUNTT` | GPU-NTT "4-step" radix | Requires log₂(n) ∈ [12, 24]. May be faster on very large n. |
| `MUL_FFT_CUFFT` | Complex FFT via cuFFT | Approximate. Good for moderate digit counts with small `LIMB_BITS`. |
| `MUL_FFT_GPUFFT` | GPU-FFT C2C (Alisah-Ozcan) | Approximate. Requires CMake ≥ 3.26. |
| `MUL_FFNT_GPUFFT` | GPU-FFT negacyclic real (FFNT) | Approximate. ~2× throughput vs cuFFT. Requires CMake ≥ 3.26. |
| `MUL_SCHOOLBOOK` | Direct O(n²) convolution | Only for tiny n_limbs (debugging / baseline). |

See [docs/backends.md](backends.md) for a detailed comparison.

---

### `MOD_REDUCTION_ALG` (default: `MOD_RED_BARRETT`)

Modular reduction algorithm used for every `modmul` and `modsq` operation.

| Value | Algorithm | Working form |
|-------|-----------|-------------|
| `MOD_RED_MONTGOMERY` | Montgomery REDC | Inputs/outputs are in Montgomery form `x·R mod N` |
| `MOD_RED_BARRETT` | Barrett reduction | Inputs/outputs are plain residues `x mod N` |

**Barrett** avoids the conversion overhead at the cost of a slightly wider NTT
buffer (`n_limbs + 1` instead of `n_limbs`). Prefer Barrett unless you have a
reason to use Montgomery.

---

## Carry normalization

After each NTT-based multiplication the coefficient array needs to be "carried"
back into canonical limb form (each limb < `2^LIMB_BITS`).

### `CARRY_NORM_ALG` (default: `CARRY_ALG_SINGLE_TILE`)

| Value | Description |
|-------|-------------|
| `CARRY_ALG_SINGLE_TILE` | One CUDA block per candidate, carries all tiles sequentially in shared memory. Simple, good occupancy on wide batches. **Recommended default.** |
| `CARRY_ALG_MULTI_TILE` | Two-phase: intra-tile carry in parallel, then inter-tile carry sequentially. Higher parallelism per candidate at the cost of two kernel launches. |
| `CARRY_ALG_SEQUENTIAL` | One thread per candidate, fully sequential. Minimal resources. Use only as a fallback or baseline. |
| `CARRY_ALG_PREFIX_SCAN` | Block-wide prefix scan (Kogge-Stone). Most parallel but requires tuning `MR_PSCAN_TILE`. |

### `MR_CARRY_TILE` (default: `32`)

Tile size (threads per block) for the `SINGLE_TILE` and `MULTI_TILE` algorithms.
Must be a multiple of 32 (warp size). Controls the shared-memory working set
per block.

### `MR_PSCAN_TILE` (default: `32`)

Threads per block for `CARRY_ALG_PREFIX_SCAN`. `32` → single-warp shuffle scan;
larger values → hierarchical block-wide scan. Must be a multiple of 32.

### `MR_CARRY_INTER_THR` (default: `32`)

Threads per block for the inter-tile phase of `CARRY_ALG_MULTI_TILE`.

---

## Kernel thread counts

These control the number of CUDA threads per block for specific kernels. All
values must be **multiples of 32** (the GPU warp size). Changing them rarely
helps without profiling; the defaults work well across most GPUs.

| Parameter | Kernel | Default |
|-----------|--------|---------|
| `MR_THR_LOAD` | `load_padded_batch` — copies & zero-pads input limbs | 256 |
| `MR_THR_PMUL` | `pmul_batch` / `psq_batch` — pointwise NTT-domain multiply | 256 |
| `MR_THR_REDUCE` | `extract_low` / `shift_right` — Montgomery REDC step | 256 |
| `MR_THR_SELECT_WIN` | `select_window_kernel` — reads exponent window power table | 256 |
| `MR_THR_CHECK` | `check_passed_kernel` / `check_equals_kernel` — MR result check | 256 |
| `MR_THR_COPY` | `bar_copy_out` (Barrett) — copies final residue to output | 128 |

---

## Subtraction

### `MR_SUB_TILE` (default: `256`)

Tile size for the conditional subtraction kernel (`cond_sub_batch`), which
ensures the result of a reduction is less than N. Must be a multiple of 32.

---

## Monitoring

### `MR_PROGRESS_INTERVAL_MS` (default: `2000`)

Minimum time in milliseconds between progress-bar updates during a GPU run.
Higher values reduce overhead; lower values give more responsive feedback.
Only relevant when the `--progress` flag is passed.

### `MR_ADVANCED_MONITOR` (default: `ON`)

When `ON`, prints detailed carry-iteration statistics after each kernel batch.
Useful for profiling carry normalization, but adds measurable overhead on large
batches. Set to `OFF` for production runs.

---

## GPU-NTT library options

These options affect the external [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)
library that is fetched automatically by CMake.

### `GPUNTT_CC89` (default: `OFF`)

Enable optimized kernel tables for **Compute Capability 8.9** (NVIDIA RTX 4090).

- `ON` → uses hand-tuned shared-memory and grid configurations for `n_power` 27
  and 28, which can significantly improve throughput on a 4090.
- `OFF` → uses generic configurations for all `n_power` values.

Set to `ON` only if your GPU is a CC 8.9 device.

### `GPUNTT_NTT_LAYOUT` (default: `PerPolynomial`)

Controls how the batch of polynomials is laid out in the NTT buffers.

| Value | Layout | Notes |
|-------|--------|-------|
| `PerPolynomial` | One polynomial (candidate) per row | **Recommended.** Better memory locality for most access patterns. |
| `PerCoefficient` | One coefficient index per row (transposed) | May benefit specific hardware memory subsystems. |

---

## Compiler flags

### `MR_MAXRREGCOUNT` (default: `0`)

Maps to `nvcc -maxrregcount N`. Limits the number of registers each CUDA thread
may use.

- `0` → no limit; the compiler decides (maximizes IPC, may reduce occupancy).
- Lower values (e.g. 64, 48) force higher thread occupancy at the cost of
  register spilling to slower local memory.

Only adjust this if you have profiled and know occupancy is the bottleneck.
