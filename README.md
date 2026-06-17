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
| `--test` | Run GMP correctness checks before benchmarking |
| `--progress` | Show live GPU progress bar |
| `--report` | Per-candidate detail report |
| `--config` | Print active build configuration and exit |
| `--bench-ops` | Benchmark individual GPU primitives |
| `--bench-ops-long` | Longer primitive benchmark |

## Project layout

```
.
├── CMakeLists.txt            # Single target: bench_mr_gpu
├── params.cmake.example      # All build parameters (copy to params.cmake)
├── example.txt               # Sample candidates in equation format
├── docs/                     # Extended documentation
└── src/
    ├── bench_mr_gpu.cu       # Entry point & driver
    ├── equation.h            # Arithmetic parser  →  mpz_t (GMP)
    ├── candidate.cuh         # NumberCandidate / GroupCandidate
    ├── miller_rabin_runner.* # GPU exponentiation pipeline
    ├── batch_mod_ctx.*       # Batched modular context (GPU buffers & tables)
    ├── reductions/           # Montgomery & Barrett reduction kernels
    ├── ops/                  # mul / carry / shift / sub GPU kernels
    └── helpers/ perf/        # Timers, micro-benchmarks, profiling tree
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
