# Integration Guide

This guide is for developers who want to call MillerRabinGPU from their own
C++/CUDA program instead of using the `bench_mr_gpu` binary.

---

## Overview of the public API

The project exposes three layers of abstraction. Use the highest one that fits
your needs:

| Layer | Entry point | Best for |
|-------|------------|---------|
| **High** | `GroupCandidate` + `EquationParser` | Strings / equation-driven inputs |
| **Mid** | `NumberCandidate` + `BatchModCtx` + `gpu_miller_rabin` | When you already have `mpz_t` or limb arrays |
| **Low** | `BatchModCtx::modmul_batch` / `modsq_batch` | Custom GPU modular arithmetic loops |

All headers are under `src/`. There is no installed library target yet — the
simplest approach is to add this repository as a CMake subdirectory or copy the
`src/` tree into your project.

---

## Adding to your CMake project

### Option A — FetchContent (recommended)

```cmake
include(FetchContent)
FetchContent_Declare(
    MillerRabinGPU
    GIT_REPOSITORY https://github.com/crazynds/MillerRabin-GPU.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(MillerRabinGPU)

# Link against the bench_mr_gpu target's include dirs manually, or
# add an INTERFACE library target in your own CMakeLists.
target_include_directories(my_app PRIVATE
    ${millerrabingpu_SOURCE_DIR}/src
)
target_link_libraries(my_app PRIVATE
    GPUNTT::ntt
    CUDA::cudart
    CUDA::cufft
    gmp
)
```

### Option B — subdirectory

```cmake
add_subdirectory(vendor/MillerRabinGPU)
target_include_directories(my_app PRIVATE vendor/MillerRabinGPU/src)
target_link_libraries(my_app PRIVATE GPUNTT::ntt CUDA::cudart CUDA::cufft gmp)
```

> **Important**: you must run CMake on MillerRabinGPU at least once so that
> `src/config.h` is generated from `src/config.h.in`. The generated file is
> needed by every other header.

---

## Layer 1 — High level: equation strings

The simplest integration path. Give each candidate as an arithmetic string;
the library handles all GMP construction and batching.

```cpp
#include "candidate.cuh"          // GroupCandidate
#include "miller_rabin_runner.cuh" // gpu_miller_rabin / gpu_miller_rabin_s1
#include "batch_mod_ctx.cuh"      // BatchModCtx

// 1. Describe candidates
GroupCandidate g;
g.label = "my_group";
g.equations.push_back("10^18001 - 25*10^1334 - 91*10^249 - 1");
g.equations.push_back("10^18001 - 52*10^16665 - 19*10^17750 - 1");

// 2. Build (evaluates equations with GMP, packs limb arrays)
g.build();   // throws std::runtime_error on bad equation or non-positive value

// 3. Test each equation
int n_limbs = g.n_limbs;
for (size_t round = 0; round < g.cands.size(); round++) {
    NumberCandidate &cand = g.cands[round];

    // Pack a single-element batch
    std::vector<uint64_t> N_batch(n_limbs, 0);
    std::vector<uint64_t> Nm1_batch(n_limbs, 0);
    std::vector<uint64_t> d_batch(n_limbs, 0);
    std::copy(cand.N_lims.begin(),   cand.N_lims.end(),   N_batch.begin());
    std::copy(cand.Nm1_lims.begin(), cand.Nm1_lims.end(), Nm1_batch.begin());
    std::copy(cand.d_lims.begin(),   cand.d_lims.end(),   d_batch.begin());

    BatchModCtx ctx(N_batch, n_limbs, /*n_batch=*/1);

    std::vector<bool> result;
    if (cand.s == 1)
        result = gpu_miller_rabin_s1(ctx, d_batch, Nm1_batch, 1,
                                     DEFAULT_WITNESSES, "test");
    else
        result = gpu_miller_rabin(ctx, d_batch, Nm1_batch, cand.s, 1,
                                  DEFAULT_WITNESSES, "test");

    if (!result[0]) {
        printf("Composite at round %zu — group eliminated.\n", round);
        break;
    }
    printf("Round %zu passed (probable prime).\n", round);
}
```

---

## Layer 2 — Mid level: mpz_t or limb arrays

Use this when you construct numbers in your own code with GMP and want to hand
them directly to the GPU without going through an equation string.

### From `mpz_t`

```cpp
#include "candidate.cuh"
#include "miller_rabin_runner.cuh"
#include "batch_mod_ctx.cuh"

// Suppose you have an mpz_t N already set somewhere
mpz_t N;
mpz_init(N);
mpz_ui_pow_ui(N, 2, 1279);   // Mersenne prime M1279 = 2^1279 - 1
mpz_sub_ui(N, N, 1);

// Build a NumberCandidate from the mpz_t
int n_limbs = limbs_for_digits((int)mpz_sizeinbase(N, 10) + 4);
NumberCandidate cand;
cand.build_from_mpz(N, n_limbs);
mpz_clear(N);

// Pack and test (same as Layer 1 from here)
std::vector<uint64_t> N_b(n_limbs), Nm1_b(n_limbs), d_b(n_limbs);
std::copy(cand.N_lims.begin(),   cand.N_lims.end(),   N_b.begin());
std::copy(cand.Nm1_lims.begin(), cand.Nm1_lims.end(), Nm1_b.begin());
std::copy(cand.d_lims.begin(),   cand.d_lims.end(),   d_b.begin());

BatchModCtx ctx(N_b, n_limbs, 1);
auto result = (cand.s == 1)
    ? gpu_miller_rabin_s1(ctx, d_b, Nm1_b, 1, DEFAULT_WITNESSES, "M1279")
    : gpu_miller_rabin   (ctx, d_b, Nm1_b, cand.s, 1, DEFAULT_WITNESSES, "M1279");

printf("%s\n", result[0] ? "probable prime" : "composite");
```

### Batching multiple candidates

The GPU is most efficient when you test many numbers at once. Pack a flat
`[n_batch × n_limbs]` array:

```cpp
int n_batch = 64;    // test 64 numbers at once
// n_limbs must be >= the largest candidate's limb count.
// Smaller candidates are zero-padded; pack_batch handles this automatically.
int n_limbs = ...;   // use the maximum across all candidates in the batch

std::vector<uint64_t> N_all(n_batch * n_limbs, 0);
std::vector<uint64_t> Nm1_all(n_batch * n_limbs, 0);
std::vector<uint64_t> d_all(n_batch * n_limbs, 0);
std::vector<int>      s_vals(n_batch);

for (int i = 0; i < n_batch; i++) {
    // ... fill your mpz_t my_N[i] ...
    NumberCandidate c;
    c.build_from_mpz(my_N[i], n_limbs);
    s_vals[i] = c.s;

    std::copy(c.N_lims.begin(),   c.N_lims.end(),   N_all.begin()   + i * n_limbs);
    std::copy(c.Nm1_lims.begin(), c.Nm1_lims.end(), Nm1_all.begin() + i * n_limbs);
    std::copy(c.d_lims.begin(),   c.d_lims.end(),   d_all.begin()   + i * n_limbs);
}

// All candidates must share the same s for the s=1 fast path.
// Use the general version when s values differ, passing the max s.
int s = *std::max_element(s_vals.begin(), s_vals.end());

BatchModCtx ctx(N_all, n_limbs, n_batch);
std::vector<bool> results = gpu_miller_rabin(
    ctx, d_all, Nm1_all, s, n_batch, DEFAULT_WITNESSES, "batch");

for (int i = 0; i < n_batch; i++)
    printf("candidate %d: %s\n", i, results[i] ? "probable prime" : "composite");
```

> **Note on `s`:** if the candidates in your batch have different `s` values,
> pass the **maximum** s to `gpu_miller_rabin`. The extra squaring rounds are
> harmless for candidates with smaller s (they simply check an already-passed
> condition).

---

## Layer 3 — Low level: raw modular arithmetic

If you need to run custom modular computations on the GPU (e.g. a custom
exponentiation loop, a different primality test), you can use `BatchModCtx`
directly.

```cpp
#include "batch_mod_ctx.cuh"

BatchModCtx ctx(N_all, n_limbs, n_batch);

// Allocate GPU buffers (use ctx.ntt.padded for the stride)
int stride = ctx.ntt.padded;
Data64 *d_A, *d_B, *d_C;
cudaMalloc(&d_A, (size_t)n_batch * stride * sizeof(Data64));
cudaMalloc(&d_B, (size_t)n_batch * stride * sizeof(Data64));
cudaMalloc(&d_C, (size_t)n_batch * stride * sizeof(Data64));

// Fill d_A and d_B with limb data in working form (see to_residue_batch)
std::vector<uint64_t> a_host = ...;
ctx.to_residue_batch(a_host, a_residue);
cudaMemcpy(d_A, a_residue.data(), ...);

// d_C = d_A * d_B  mod N  (for every candidate in the batch)
ctx.modmul_batch(d_A, d_B, d_C);

// d_C = d_A^2  mod N
ctx.modsq_batch(d_A, d_C);

// Read result back to host limbs
std::vector<uint64_t> result_host;
ctx.from_residue_batch(d_C, result_host);

cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
```

### Key `BatchModCtx` members

| Member | Type | Description |
|--------|------|-------------|
| `ntt` | `Multiplier` | The NTT/FFT backend object. Access `ntt.n_limbs`, `ntt.padded`, `ntt.n_batch`. |
| `n_limbs` | `int` | Limbs per candidate (read-only after construction). |
| `n_batch` | `int` | Candidates in the batch (read-only after construction). |
| `d_N` | `Data64*` | Device pointer to flat `[n_batch × padded]` limb arrays of N. |
| `device_id` | `int` | Which GPU this context was created on. |
| `perf_enabled` | `bool` | Set to `true` to enable the profiling tree. |

### Working form

The internal representation depends on `MOD_REDUCTION_ALG`:

- **Montgomery** (`MOD_RED_MONTGOMERY`): values are stored as `x · R mod N`
  where `R = 2^(n_limbs × LIMB_BITS)`. Use `to_residue_batch` to convert
  ordinary limb arrays into Montgomery form before passing them to GPU kernels,
  and `from_residue_batch` to convert back.
- **Barrett** (`MOD_RED_BARRETT`): values are plain residues `x mod N`.
  `to_residue_batch` and `from_residue_batch` are still the correct conversion
  functions regardless of backend.

---

## Custom witness sets

Pass any `std::vector<uint32_t>` as the witness list instead of
`DEFAULT_WITNESSES`:

```cpp
// Deterministic for all N < 3,317,044,064,679,887,385,961,981
std::vector<uint32_t> my_witnesses = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

auto results = gpu_miller_rabin(ctx, d_all, Nm1_all, s, n_batch,
                                my_witnesses, "label");
```

The default set (`DEFAULT_WITNESSES` in `miller_rabin_runner.cuh`) uses 16
witnesses: `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}`,
which is more than enough for numbers of any practical size.

---

## Multi-GPU

`BatchModCtx` is tied to a single GPU at construction time. To use multiple
GPUs, create one `BatchModCtx` per device and split the batch:

```cpp
int n_devices;
cudaGetDeviceCount(&n_devices);

std::vector<BatchModCtx *> ctxs;
for (int dev = 0; dev < n_devices; dev++) {
    // Each device gets its own slice of the N_all array
    int slice_start = dev * slice_size;
    std::vector<uint64_t> N_slice(N_all.begin() + slice_start * n_limbs,
                                   N_all.begin() + (slice_start + slice_size) * n_limbs);
    ctxs.push_back(new BatchModCtx(N_slice, n_limbs, slice_size, dev));
}

// Launch on each device (possibly in parallel with std::thread)
for (int dev = 0; dev < n_devices; dev++) {
    cudaSetDevice(dev);
    // ... gpu_miller_rabin(*ctxs[dev], ...) ...
}
```

---

## Error handling

All errors from the GMP/parsing layer throw `std::runtime_error`. CUDA errors
inside `BatchModCtx` and the runner are checked with a `CU()` macro that also
throws `std::runtime_error`. Wrap calls in a `try/catch` block:

```cpp
try {
    g.build();
    BatchModCtx ctx(N_all, n_limbs, n_batch);
    auto results = gpu_miller_rabin(ctx, d_all, Nm1_all, s, n_batch,
                                    DEFAULT_WITNESSES, "label");
    // use results...
} catch (const std::runtime_error &e) {
    fprintf(stderr, "Error: %s\n", e.what());
}
```

---

## Full minimal example

```cpp
// my_prime_check.cu
#include <cstdio>
#include <vector>
#include <string>
#include "candidate.cuh"
#include "miller_rabin_runner.cuh"
#include "batch_mod_ctx.cuh"

int main()
{
    // Equations to test — can be anything the parser supports
    std::vector<std::string> equations = {
        "2^1279 - 1",          // Mersenne prime
        "10^999 + 7",          // random large number (likely composite)
        "2^2203 - 1",          // another Mersenne prime
    };

    // Build candidates
    std::vector<NumberCandidate> cands;
    int n_limbs = 0;
    for (auto &eq : equations) {
        mpz_t val;
        mpz_init(val);
        EquationParser::eval(eq, val);
        int d = (int)mpz_sizeinbase(val, 10);
        if (d > n_limbs) n_limbs = limbs_for_digits(d + 4);
        mpz_clear(val);
    }
    for (auto &eq : equations) {
        mpz_t val;
        mpz_init(val);
        EquationParser::eval(eq, val);
        NumberCandidate c;
        c.build_from_mpz(val, n_limbs);
        mpz_clear(val);
        cands.push_back(c);
    }

    // Pack flat batch
    int n_batch = (int)cands.size();
    std::vector<uint64_t> N_all(n_batch * n_limbs, 0);
    std::vector<uint64_t> Nm1_all(n_batch * n_limbs, 0);
    std::vector<uint64_t> d_all(n_batch * n_limbs, 0);
    int max_s = 0;
    for (int i = 0; i < n_batch; i++) {
        std::copy(cands[i].N_lims.begin(),   cands[i].N_lims.end(),   N_all.begin()   + i * n_limbs);
        std::copy(cands[i].Nm1_lims.begin(), cands[i].Nm1_lims.end(), Nm1_all.begin() + i * n_limbs);
        std::copy(cands[i].d_lims.begin(),   cands[i].d_lims.end(),   d_all.begin()   + i * n_limbs);
        if (cands[i].s > max_s) max_s = cands[i].s;
    }

    // Run on GPU
    BatchModCtx ctx(N_all, n_limbs, n_batch);
    auto results = gpu_miller_rabin(ctx, d_all, Nm1_all, max_s, n_batch,
                                    DEFAULT_WITNESSES, "demo");

    for (int i = 0; i < n_batch; i++)
        printf("%-25s  →  %s\n", equations[i].c_str(),
               results[i] ? "probable prime" : "composite");
    return 0;
}
```

Compile with:

```sh
nvcc -std=c++17 -O3 --use_fast_math \
     -I/path/to/MillerRabinGPU/src \
     -I/path/to/gpuntt/include \
     my_prime_check.cu \
     /path/to/MillerRabinGPU/build/CMakeFiles/bench_mr_gpu.dir/src/*.o \
     -lgmp -lcufft -lcudart \
     -o my_prime_check
```

Or, preferably, add it via CMake as shown in [Adding to your CMake project](#adding-to-your-cmake-project).
