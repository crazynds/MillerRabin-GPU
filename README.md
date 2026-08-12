<div align="center">

# MillerRabinGPU

**GPU-accelerated Miller–Rabin primality test for very large integers**

[![CUDA](https://img.shields.io/badge/CUDA-11%2B-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-blue?logo=cmake)](https://cmake.org)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/17)

[Quick Start](#quick-start) · [Input Format](#input-format) · [Documentation](#documentation) · [Architecture](#how-it-works)

</div>

Tests thousands-of-digit prime candidates in large GPU batches using FFT-based
big-integer arithmetic with Montgomery or Barrett modular reduction. Candidates
are expressed as arbitrary arithmetic equations — no fixed number format required.

```
1: 10^18001 - 25*10^1334 - 91*10^249 - 1
1: 10^18001 - 52*10^16665 - 19*10^17750 - 1
2: 2^74207281 - 1
```

## Quick start

```sh
# Install dependencies (Ubuntu/Debian)
sudo apt install libgmp-dev gcc-11 g++-11

# Clone
git clone https://github.com/crazynds/MillerRabin-GPU
cd MillerRabin-GPU
cp params.cmake.example params.cmake

# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run
./build/bench_mr_gpu --progress example.txt
```

> **Requirements:** CUDA ≥ 11 · CMake ≥ 3.18 · GMP (`libgmp-dev`) · CUDA-capable GPU  
> The [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) library is fetched automatically by CMake.  
> → Full guide: [docs/building.md](docs/building.md)

## How it works

For each candidate `N`, the test decomposes `N − 1 = 2ˢ · d` and runs standard
Miller–Rabin rounds for a set of witnesses — entirely on the GPU using windowed
modular exponentiation. The heavy lifting is big-integer multiplication in the
**frequency domain** (FFT/NTT), followed by Montgomery or Barrett modular reduction.

The default backend (`MUL_MERGE_GPUNTT`) handles numbers up to ~650 million
decimal digits. All parameters — batch size, multiplication algorithm, reduction
method, kernel thread counts — are tuned via a single [`params.cmake`](params.cmake.example) file.

→ Deep dive: [docs/architecture.md](docs/architecture.md) · [docs/backends.md](docs/backends.md)

## Input format

Each line is one equation, with an optional **group ID** prefix:

```
[group_id:] equation
```

| Part | Description |
|------|-------------|
| `group_id` | Optional label. Lines with the same ID are tested as a group: if one is composite, the rest are **skipped**. |
| `equation` | Arithmetic expression: `+ - * / % ^` and `( )`. Numbers are arbitrary-precision (GMP). |

```
# Group: both N and its digit-reversed twin must be prime
1: 10^18001 - 25*10^1334 - 91*10^249 - 1
1: 10^18001 - 52*10^16665 - 19*10^17750 - 1

# Standalone (no group)
2^1279 - 1

# Giant literal — GMP reads it exactly, no overflow
314159265358979323846...
```

→ Full grammar and group semantics: [docs/input-format.md](docs/input-format.md)

## Usage

```sh
./build/bench_mr_gpu [options] <input.txt>
```

| Option | Description |
|--------|-------------|
| `--test` | Run GMP-checked correctness tests before the benchmark |
| `--report` | Print a per-candidate detail report |
| `--progress` | Show a live GPU progress bar |
| `--config` | Print the active build configuration and exit |
| `--bench-ops` | Benchmark the individual GPU primitives |
| `--bench-ops-long` | Longer/more thorough primitive benchmark |
| `--help`, `-h` | Show the full help and exit |

### Running on the CPU instead

Same group semantics, using GMP `mpz_probab_prime_p`:

| Option | Description |
|--------|-------------|
| `--cpu` | One candidate at a time, no batching |
| `--cpu-parallel` | Spread across all hardware threads |
| `--threads N`, `-j N` | Spread across N threads |

### `--bench-compare`

Self-benchmark of GPU against CPU on a freshly generated, small-factor-free
candidate. Ignores `<input.txt>`. Without `--do`, the GPU runs once and the CPU
is swept across every thread count from 1 to the machine's hardware concurrency.

| Option | Default | Description |
|--------|---------|-------------|
| `--digits N` | 100000 | Decimal digits of the test candidate |
| `--timeout S` | 30 | Seconds per run for the throughput phase |
| `--single-iters K` | 10 | Latency-phase repetitions per side |
| `--skip-phase-1` | | Skip the throughput phase |
| `--skip-phase-2` | | Skip the single-candidate latency phase |
| `--do=WHO` | | Restrict to one side and print each phase as it finishes: `--do=gpu`, or `--do=Ncpu` for CPU with exactly N threads (`--do=1cpu`, `--do=8cpu`). Useful for firing one job per side. |

## Project layout

```
.
├── CMakeLists.txt            # Single target: bench_mr_gpu
├── params.cmake.example      # All build parameters (copy to params.cmake)
├── example.txt               # Sample candidates in equation format
├── docs/                     # Extended documentation
└── src/
    ├── main.cu               # Entry point
    ├── cli/                  # Argument parsing, --help, --config
    ├── driver/               # Equation parser, candidates, round driver, CPU path
    ├── mr/                   # Miller-Rabin GPU exponentiation pipeline
    ├── mod/                  # Batched modular context + reductions/ (Montgomery, Barrett)
    ├── ntt/                  # Vendored/fused NTT & INTT kernels, Shoup butterflies
    ├── ops/                  # mul/ carry/ shift/ sub GPU kernels
    ├── tests/                # GMP-checked correctness tests (--test)
    ├── bench/                # --bench-ops, --bench-compare
    └── perf/ util/           # Profiling tree, timers, helpers
```

## Documentation

| | |
|---|---|
| [📦 Building](docs/building.md) | System requirements, dependencies, build options, troubleshooting |
| [🔌 Integration](docs/integration.md) | Using as a library — API layers, CMake setup, code examples |
| [📄 Input format](docs/input-format.md) | Equation grammar, group semantics, large-number handling |
| [⚙️ Configuration](docs/configuration.md) | Every `params.cmake` parameter explained in detail |
| [🧮 Backends](docs/backends.md) | Multiplication backends — differences, limits, selection guide |
| [🏗️ Architecture](docs/architecture.md) | GPU pipeline, FFT/NTT multiplication, modular reduction internals |
