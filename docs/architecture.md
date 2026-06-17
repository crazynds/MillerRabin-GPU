# Architecture

This document explains how the GPU pipeline works internally — from reading a
candidate equation to producing a primality verdict.

## High-level flow

```
Input file
    │
    ▼
parse_input()           ← bench_mr_gpu.cu
    │  groups: label + equations[]
    ▼
GroupCandidate::build() ← candidate.cuh
    │  EquationParser (equation.h) → mpz_t (GMP)
    │  N, N-1, d = (N-1)/2^s packed into limb arrays
    ▼
Round loop (round = 0, 1, 2, …)
    │  alive groups with a candidate at this round are batched
    ▼
test_batch()            ← bench_mr_gpu.cu
    │
    ├─ BatchModCtx(N_batch, n_limbs, bsz)   ← batch_mod_ctx.cu
    │     pre-computes Montgomery/Barrett tables on GPU
    │
    └─ gpu_miller_rabin() or gpu_miller_rabin_s1()   ← miller_rabin_runner.cu
          for each witness:
              mod_exp(base, d) → r
              for i in 0..s-1: r = r² mod N, check r == N-1
          returns bool[] (passed/failed) per candidate
```

## Number representation

Every big integer is stored as an array of `n_limbs` unsigned 64-bit values,
each holding one "digit" in base `2^LIMB_BITS` (default 16 bits → each limb
stores values 0–65535).

```
N = limbs[0]  +  limbs[1] × 2^16  +  limbs[2] × 2^32  +  …
```

A batch of `n_batch` numbers is stored as a flat array of size
`n_batch × n_limbs` in row-major order (candidate-first):

```
buf[i * n_limbs + j]  =  limb j of candidate i
```

This layout lets a single GPU kernel processcandidates simultaneously by batch.

## Modular arithmetic context (`BatchModCtx`)

`BatchModCtx` owns all GPU memory for one batch and pre-computes the tables
needed by the chosen reduction backend.

### Montgomery mode (`MOD_RED_MONTGOMERY`)

Montgomery reduction avoids explicit division by working in "Montgomery form":
instead of computing `x mod N`, all values are stored as `x·R mod N` where
`R = 2^(n_limbs × LIMB_BITS)`. Multiplying two Montgomery-form values and
running REDC produces another Montgomery-form value — no expensive division.

Pre-computed per batch:
- `N'` (the Montgomery inverse): `N' = -N^{-1} mod R`
- `R² mod N` (for converting ordinary integers into Montgomery form)

### Barrett mode (`MOD_RED_BARRETT`)

Barrett reduction replaces modular division with a multiply-shift:
precompute `μ = floor(R² / N)`, then `x mod N ≈ x - N × floor(x × μ / R²)`.
The approximation is exact for `x < N²` (which is always the case after a
single multiplication).

Pre-computed per batch:
- `μ_i = floor(b^{2k} / N_i)` for each candidate i, stored as a limb array

## Multiplication pipeline

Big-integer multiplication is done in the **frequency domain** using a Fast
Fourier Transform. The idea is the same as classic FFT-based polynomial
multiplication — a convolution in the time domain becomes a pointwise product
in the frequency domain:

> **A × B = IFFT( FFT(A) · FFT(B) )**

The difference from a floating-point FFT is that this project uses an **NTT
(Number Theoretic Transform)** for the exact backends. An NTT is simply an FFT
evaluated over a finite field (integers modulo a prime) instead of complex
numbers, so every coefficient stays an exact integer with zero rounding error.
The FFT backends (`MUL_FFT_CUFFT`, `MUL_FFT_GPUFFT`, `MUL_FFNT_GPUFFT`) use
the classical complex-valued FFT (double precision), which is faster but
approximate.

Each modular multiplication (`mont_mul` or `bar_mul`) follows this sequence:

```
1. Forward FFT/NTT of A  →  A_freq
2. Forward FFT/NTT of B  →  B_freq   (or reuse a pre-computed B_freq for N' / μ)
3. Pointwise product:  C_freq[i] = A_freq[i] × B_freq[i]
4. Inverse FFT/NTT of C_freq  →  C_raw   (un-normalized, coefficients may be large)
5. Carry normalization  →  C_limbs   (canonical form, each limb < 2^LIMB_BITS)
6. Reduction step (REDC or Barrett)  →  result mod N
```

Steps 1–4 are handled by the `Multiplier` class (the pluggable FFT/NTT
backend). Steps 5–6 are handled by `carry_norm.cu` and the relevant file in
`reductions/`.

### Carry normalization

After the inverse FFT/NTT, each coefficient may hold a value much larger than
`2^LIMB_BITS`. The carry pass sweeps left-to-right:

```
for i in 0..n_limbs:
    carry  = limbs[i] >> LIMB_BITS
    limbs[i] &= LIMB_MASK
    limbs[i+1] += carry
```

Four strategies exist (controlled by `CARRY_NORM_ALG` in `params.cmake`);
see [configuration.md](configuration.md) for details.

> The same carry step is needed after both NTT and complex-FFT backends.
> The NTT case produces exact integer coefficients; the FFT case produces
> floating-point values that are rounded to the nearest integer first.

## Modular exponentiation

`miller_rabin_runner.cu` implements **left-to-right windowed exponentiation**:

1. **Table pre-compute**: for each candidate, build a power table
   `T[0] = 1,  T[1] = base,  T[2] = base²,  …,  T[W-1] = base^(W-1)`
   where `W = 2^MR_WINDOW_BITS`. This is done with `W-1` modular multiplications.

2. **Exponentiation loop**: scan the exponent `d` from MSB to LSB in chunks of
   `MR_WINDOW_BITS` bits. For each window value `w`:
   - Square the accumulator `MR_WINDOW_BITS` times.
   - Multiply by `T[w]` (one look-up + one multiplication).

3. **Miller–Rabin check**: after computing `r = a^d mod N`:
   - If `r == 1` or `r == N-1` → witness passes.
   - Otherwise, square `r` up to `s-1` times; if any squaring yields `N-1` → passes.
   - If none → candidate is **composite**.

### s=1 fast path

Many prime candidates of the form `10^a - … - 1` have `N ≡ 3 (mod 4)`, meaning
`s = 1` (i.e. `N-1 = 2d`). In this case there are no squaring rounds to check
after the initial exponentiation — a separate optimized kernel
`gpu_miller_rabin_s1` is dispatched.

## Batch testing and group short-circuiting

The driver processes candidates in rounds:

```
Round 1: batch all "equation 0" of every alive group → GPU → update alive[]
Round 2: batch all "equation 1" of surviving groups  → GPU → update alive[]
…
```

Within each round, the active groups are **sorted by `n_limbs` descending**
before being sliced into `BATCH_SIZE` chunks. This clustering means that
candidates of the same size are most likely to end up in the same chunk,
keeping the FFT/NTT size uniform (optimal).

When a chunk contains groups of different `n_limbs` — because candidates of
genuinely different sizes happen to share a chunk — the batch is normalised to
the **largest `n_limbs` in that chunk**. Smaller candidates are zero-padded to
that width: `pack_batch` zero-initialises the flat output buffer before copying
each candidate's limbs, so the extra high limbs remain zero. Zero-padding is
semantically safe because `BatchModCtx` precomputes its tables per-candidate
from each candidate's own `N` value, and the extra zero limbs do not change
the number's value.

## Equation parsing and GMP construction

`equation.h` implements a recursive-descent parser that evaluates an arithmetic
expression string into a GMP `mpz_t`. All intermediate values are GMP integers
— there is no truncation or overflow at any step. A 100 000-digit literal is
read verbatim by `mpz_set_str`; `10^99999` is computed exactly by
`mpz_pow_ui`.

After evaluation, `NumberCandidate::build_from_mpz()`:
1. Computes `N-1` and finds `s` (the 2-adic valuation of `N-1`).
2. Computes `d = (N-1) / 2^s`.
3. Converts `N`, `N-1`, and `d` to limb arrays via `mpz_to_limbs_vec()`.

The limb width `n_limbs` is chosen as `limbs_for_digits(max_digits + 4)` — a
small padding that ensures FFT/NTT size constraints are never tight.

## Performance profiling

A lightweight in-process profiling tree (`src/perf/`) records time spent in each
sub-step (FFT/NTT forward, pointwise multiply, carry, REDC, …). The tree is printed
at the end of each batch when `--report` is passed. The `MR_ADVANCED_MONITOR`
build flag additionally prints per-iteration carry statistics.
