# Building MillerRabinGPU

## Requirements summary

| Dependency | Minimum version | How to get it |
|------------|-----------------|---------------|
| NVIDIA GPU | Compute Capability ≥ 3.5 | Hardware |
| CUDA Toolkit | 11.x (12.x works with `--allow-unsupported-compiler`) | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| CMake | 3.18 (3.26 for GPU-FFT backends) | Package manager or [cmake.org](https://cmake.org/download/) |
| GCC / G++ | ≤ 11 for CUDA 11 | Package manager |
| GMP | any recent | `libgmp-dev` via package manager |
| Git | any | Package manager (needed for FetchContent) |
| GPU-NTT | auto | Fetched automatically by CMake |
| GPU-FFT | auto | Fetched automatically by CMake (only if using a GPU-FFT backend) |

## System dependencies

### Ubuntu / Debian

```sh
sudo apt update
sudo apt install build-essential cmake git libgmp-dev

# GCC 11 (recommended for CUDA 11)
sudo apt install gcc-11 g++-11
```

### Fedora / RHEL

```sh
sudo dnf install cmake git gmp-devel gcc gcc-c++

# GCC 11
sudo dnf install gcc-toolset-11   # RHEL/CentOS stream
# or
sudo dnf install gcc11 gcc11-c++  # Fedora
```

### Arch Linux

```sh
sudo pacman -S cmake git gmp base-devel
# GCC 11 is available in the AUR: gcc11
```

## CUDA Toolkit

If not already installed, grab the installer for your OS and CUDA version from:

> https://developer.nvidia.com/cuda-downloads

Verify after installing:

```sh
nvcc --version
nvidia-smi
```

### CUDA / GCC compatibility

CUDA 11.x officially supports up to GCC 11. The build system handles this
automatically:

- If `gcc-11` is found on `PATH`, it is used as the host compiler.
- If not found, `-allow-unsupported-compiler` is passed to `nvcc` so you can
  build with a newer GCC (may produce warnings but generally works).

| CUDA version | Max supported GCC |
|--------------|-------------------|
| 11.x | GCC 11 |
| 12.x | GCC 12 |
| 12.4+ | GCC 13 |

## CMake

Minimum required version is **3.18**. Check your version:

```sh
cmake --version
```

If it is older than 3.18, upgrade:

```sh
# Ubuntu — install from Kitware's apt repo
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt update && sudo apt install cmake

# Or download a binary directly from cmake.org/download
```

> If you plan to use the `MUL_FFT_GPUFFT` or `MUL_FFNT_GPUFFT` backends,
> you need **CMake ≥ 3.26** (required by the GPU-FFT library).

## GMP (GNU Multiple Precision Arithmetic Library)

GMP is used on the CPU side to construct candidates from equation strings and
to run correctness tests (`--test`). It is **not** used on the GPU.

```sh
# Ubuntu / Debian
sudo apt install libgmp-dev

# Fedora / RHEL
sudo dnf install gmp-devel

# Arch
sudo pacman -S gmp

# macOS (Homebrew)
brew install gmp
```

CMake will find GMP automatically via `find_library(gmp)` and `find_path(gmp.h)`.
If GMP is installed in a non-standard location, point CMake to it:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/gmp
```

## Automatically fetched libraries

The following libraries are downloaded and compiled by CMake via `FetchContent`
— **you do not need to install them manually**.

### GPU-NTT (always required)

> https://github.com/Alisah-Ozcan/GPU-NTT

Provides the GPU number-theoretic transform used by the NTT multiplication
backends. Fetched on first `cmake` configure; subsequent builds use the cached
copy.

If the machine has no internet access, clone the repo and point CMake to it:

```sh
git clone https://github.com/Alisah-Ozcan/GPU-NTT /opt/gpuntt
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GPUNTT=/opt/gpuntt
```

### GPU-FFT (only for `MUL_FFT_GPUFFT` / `MUL_FFNT_GPUFFT`)

> https://github.com/Alisah-Ozcan/GPU-FFT

Required only when `MUL_ALG` is set to `MUL_FFT_GPUFFT` or `MUL_FFNT_GPUFFT`
in `params.cmake`. Not downloaded otherwise.

Same offline override:

```sh
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_GPUFFT=/opt/gpufft
```

## Build steps

### First time

```sh
# 1. Copy the parameter template
cp params.cmake.example params.cmake

# 2. (Optional) edit params.cmake to your liking
#    e.g. change MUL_ALG, LIMB_BITS, MR_BATCH_SIZE

# 3. Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 4. Compile (use -j to parallelize)
cmake --build build -j$(nproc)
```

The binary is produced at `build/bench_mr_gpu`.

### Subsequent builds

After changing `params.cmake` or any source file:

```sh
cmake --build build -j$(nproc)
```

CMake detects the changes automatically. A re-configure (`cmake -S . -B build`)
is only needed if you add/remove files or change CMake options.

### Clean rebuild

```sh
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Build options

These can be passed on the `cmake` command line to override `params.cmake`
values temporarily without editing the file:

```sh
# Override the multiplication algorithm
cmake -S . -B build -DMUL_ALG=MUL_FFT_CUFFT

# Override batch size
cmake -S . -B build -DMR_BATCH_SIZE=128

# Release with debug symbols
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## GPU architecture

The project uses `CMAKE_CUDA_ARCHITECTURES native`, which means nvcc
auto-detects the GPU(s) present on the build machine and compiles optimized
code for those architectures. The binary will **only run on GPUs of the same
compute capability** (or newer in most cases).

To build for a specific architecture explicitly:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 4090 (CC 8.9)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3090 (CC 8.6)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80   # A100 (CC 8.0)
```

To build for multiple architectures at once (larger binary):

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

## Verifying the build

```sh
# Print the active build configuration
./build/bench_mr_gpu --config

# Run the built-in GMP correctness tests
./build/bench_mr_gpu --test example.txt

# Benchmark the GPU primitives
./build/bench_mr_gpu --bench-ops
```

## Troubleshooting

### `nvcc fatal: Unsupported gpu architecture 'native'`

Your CMake version is too old (< 3.18). Upgrade CMake.

### `error: #error "MUL_ALG not defined"`

`src/config.h` was not generated. Run `cmake -S . -B build` to regenerate it.
This happens if you copied source files without running CMake first.

### GMP not found

```
CMake Error: Could not find GMP (libgmp). Install libgmp-dev.
```

Install `libgmp-dev` (or equivalent) and re-run `cmake -S . -B build`.

### `gcc: error: unrecognized command-line option '-allow-unsupported-compiler'`

This flag is for `nvcc`, not `gcc`. It means CMake is incorrectly passing nvcc
flags to gcc. Check that `CMAKE_CUDA_HOST_COMPILER` is set correctly and that
you are using CMake ≥ 3.18.

### Out of memory (OOM) during compilation

NTT kernel compilation is memory-intensive. Reduce parallel jobs:

```sh
cmake --build build -j2
```

### CUDA OOM at runtime

Reduce `MR_BATCH_SIZE` in `params.cmake` and rebuild.

### `FetchContent` fails (no internet)

See the [offline instructions](#automatically-fetched-libraries) above to point
CMake at a local clone of GPU-NTT and GPU-FFT.
