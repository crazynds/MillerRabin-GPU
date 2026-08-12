cmake_minimum_required(VERSION 3.18)

# ── User parameters ───────────────────────────────────────────────────────────
include(${CMAKE_CURRENT_SOURCE_DIR}/params.cmake)

# CUDA 11.x supports GCC <= 11. If the system default is newer, try gcc-11.
find_program(GCC11 gcc-11)
if(GCC11)
    set(CMAKE_CUDA_HOST_COMPILER "${GCC11}")
    set(CMAKE_C_COMPILER         "${GCC11}")
    find_program(GPP11 g++-11)
    if(GPP11)
        set(CMAKE_CXX_COMPILER "${GPP11}")
    endif()
    message(STATUS "CUDA host compiler forced to ${GCC11} (system GCC may be > 11)")
else()
    # gcc-11 not found — let nvcc use the unsupported version
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    message(STATUS "gcc-11 not found; using -allow-unsupported-compiler")
endif()

project(MillerRabinGPU LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# ── Optimization flags ────────────────────────────────────────────────────────
# Host (GCC): O3 + native CPU architecture
set(CMAKE_C_FLAGS_RELEASE   "-O3 -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
# Ensure O3 even without an explicit -DCMAKE_BUILD_TYPE=Release
add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:-O3>
    $<$<COMPILE_LANGUAGE:C>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
)
# NVCC: O3 + fast math + automatic architecture detection via CMAKE_CUDA_ARCHITECTURES
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Per-thread register limit (params.cmake: MR_MAXRREGCOUNT, 0 = no limit)
if(DEFINED MR_MAXRREGCOUNT AND NOT MR_MAXRREGCOUNT EQUAL 0)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=${MR_MAXRREGCOUNT}")
    message(STATUS "CUDA -maxrregcount=${MR_MAXRREGCOUNT}")
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# GMP
find_library(GMP_LIB gmp REQUIRED)
find_path(GMP_INCLUDE gmp.h REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── Multiplication backend selection ──────────────────────────────────────────
# Multiplication algorithm (params.cmake → MUL_ALG): compile only the .cu of the
# selected class. SCHOOLBOOK and NTT_MERGE use the "merge" class; only NTT_4STEP
# switches to ntt_4step.cu. carry_norm.cu is backend-agnostic (defines Multiplier::).
if(MUL_ALG STREQUAL "MUL_4STEP_GPUNTT")
    set(MUL_BACKEND_SRC ${SRC_DIR}/ops/mul/ntt_4step.cu)
elseif(MUL_ALG STREQUAL "MUL_FFT_CUFFT")
    set(MUL_BACKEND_SRC ${SRC_DIR}/ops/mul/fft_cufft.cu)
elseif(MUL_ALG STREQUAL "MUL_FFT_GPUFFT")
    set(MUL_BACKEND_SRC ${SRC_DIR}/ops/mul/fft_gpufft.cu)
elseif(MUL_ALG STREQUAL "MUL_FFNT_GPUFFT")
    set(MUL_BACKEND_SRC ${SRC_DIR}/ops/mul/fft_ffnt.cu)
else()
    set(MUL_BACKEND_SRC
        ${SRC_DIR}/ops/mul/ntt_merge.cu
        ${SRC_DIR}/ntt/ntt_merge_intt_fused.cu
        ${SRC_DIR}/ntt/ntt_merge_ntt_fused.cu)
endif()
message(STATUS "MUL_ALG = ${MUL_ALG} → ${MUL_BACKEND_SRC}")

# The GPU-FFT library (Alisah-Ozcan/GPU-FFT) is only needed by its FFT backends.
if(MUL_ALG STREQUAL "MUL_FFT_GPUFFT" OR MUL_ALG STREQUAL "MUL_FFNT_GPUFFT")
    set(MR_USE_GPUFFT ON)
endif()

# Generate config.h from the template + params.cmake
if(MR_ADVANCED_MONITOR)
    set(MR_ADVANCED_MONITOR 1)
else()
    unset(MR_ADVANCED_MONITOR)
endif()
if(MR_NTT_FUSED_PMUL)
    set(MR_NTT_FUSED_PMUL 1)
else()
    unset(MR_NTT_FUSED_PMUL)
endif()
if(MR_NTT_FUSED_PAD)
    set(MR_NTT_FUSED_PAD 1)
else()
    unset(MR_NTT_FUSED_PAD)
endif()
# 32-bit limb storage. Requires an integer backend (the FFT ones store limbs as
# double) and LIMB_BITS <= 32. Montgomery additionally needs a carry algorithm
# that folds the addend into its load — its REDC adds raw INTT coefficients
# (~2^58) to T, which must never be parked in a 32-bit limb slot.
if(MR_LIMB32
   AND (MUL_ALG STREQUAL "MUL_MERGE_GPUNTT" OR MUL_ALG STREQUAL "MUL_4STEP_GPUNTT"
        OR MUL_ALG STREQUAL "MUL_SCHOOLBOOK")
   AND LIMB_BITS LESS_EQUAL 32
   AND (MOD_REDUCTION_ALG STREQUAL "MOD_RED_BARRETT"
        OR NOT CARRY_NORM_ALG STREQUAL "CARRY_ALG_PREFIX_SCAN"))
    set(MR_LIMB32 1)
else()
    unset(MR_LIMB32)
endif()
# Shoup (precomputed-quotient) butterflies. Only the locally-vendored fused NTT/INTT
# kernels carry them, so it needs the merge backend; the NTT prime must be < 2^63.
if(MR_NTT_SHOUP AND MUL_ALG STREQUAL "MUL_MERGE_GPUNTT")
    set(MR_NTT_SHOUP 1)
else()
    unset(MR_NTT_SHOUP)
endif()
# Harvey lazy reduction: coefficients ride between butterflies in [0,2p)/[0,4p)
# instead of [0,p), so each butterfly needs one conditional subtract instead of
# three. Rides on the Shoup butterflies and needs 4p to fit in 64 bits.
if(MR_NTT_LAZY AND MR_NTT_SHOUP AND MUL_ALG STREQUAL "MUL_MERGE_GPUNTT")
    set(MR_NTT_LAZY 1)
else()
    unset(MR_NTT_LAZY)
endif()
# Compute q*p by shift/add using the merge prime's sparse bit pattern, moving the
# work off the multiply pipe. Needs the Shoup butterflies and that exact prime,
# which shoup_table_build verifies at setup.
if(MR_NTT_SPARSE_P AND MR_NTT_SHOUP AND MUL_ALG STREQUAL "MUL_MERGE_GPUNTT")
    set(MR_NTT_SPARSE_P 1)
else()
    unset(MR_NTT_SPARSE_P)
endif()
# The fused shift gather rides on the same fused forward-NTT kernel as the pad,
# and Barrett is the only reduction that shifts before transforming.
if(MR_NTT_FUSED_SHIFT AND MR_NTT_FUSED_PAD AND MUL_ALG STREQUAL "MUL_MERGE_GPUNTT"
   AND MOD_REDUCTION_ALG STREQUAL "MOD_RED_BARRETT")
    set(MR_NTT_FUSED_SHIFT 1)
else()
    unset(MR_NTT_FUSED_SHIFT)
endif()
configure_file(
    ${SRC_DIR}/config.h.in
    ${SRC_DIR}/config.h
)

# ── GPU-NTT library (Alisah-Ozcan/GPU-NTT) ────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    GPUNTT
    GIT_REPOSITORY https://github.com/Alisah-Ozcan/GPU-NTT.git
    GIT_TAG        main
    UPDATE_DISCONNECTED TRUE
)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/gpuntt_install" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(GPUNTT)

if(NOT TARGET GPUNTT::ntt)
    find_package(GPUNTT REQUIRED
        PATHS "${CMAKE_BINARY_DIR}/gpuntt_install/lib/cmake/GPUNTT-1.0"
        NO_DEFAULT_PATH)
endif()

# GPU-FFT (Alisah-Ozcan/GPU-FFT) — only when one of its FFT backends is selected.
# NOTE: GPU-FFT requires CMake >= 3.26; if your cmake is older, upgrade it or use
# the NTT/cuFFT backends.
if(MR_USE_GPUFFT)
    FetchContent_Declare(
        GPUFFT
        GIT_REPOSITORY https://github.com/Alisah-Ozcan/GPU-FFT.git
        GIT_TAG        main
        UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(GPUFFT)
    if(NOT TARGET GPUFFT::fft)
        find_package(GPUFFT REQUIRED)
    endif()
endif()

# ── Main binary: bench_mr_gpu ─────────────────────────────────────────────────
add_executable(bench_mr_gpu
    ${SRC_DIR}/main.cu
    ${SRC_DIR}/cli/args.cu
    ${SRC_DIR}/cli/usage.cu
    ${SRC_DIR}/cli/config_print.cu
    ${SRC_DIR}/driver/round_driver.cu
    ${SRC_DIR}/bench/bench_ops.cu
    ${SRC_DIR}/bench/bench_ops_gmp.cu
    ${SRC_DIR}/bench/bench_ops_gpu.cu
    ${SRC_DIR}/mr/miller_rabin_runner.cu
    ${SRC_DIR}/driver/input_parser.cu
    ${SRC_DIR}/driver/cpu_runner.cu
    ${SRC_DIR}/mr/mr_kernels.cu
    ${SRC_DIR}/mr/witness_buffers.cu
    ${SRC_DIR}/mr/window_exp_loop.cu
    ${SRC_DIR}/perf/perf_report.cu
    ${SRC_DIR}/tests/test_modular_ops.cu
    ${SRC_DIR}/tests/test_known_primes.cu
    ${SRC_DIR}/tests/test_general_s.cu
    ${SRC_DIR}/tests/test_s1_nextprime.cu
    ${SRC_DIR}/bench/bench_prime_finder.cu
    ${SRC_DIR}/bench/cpu_witness.cu
    ${SRC_DIR}/bench/throughput_bench.cu
    ${SRC_DIR}/bench/single_bench.cu
    ${SRC_DIR}/bench/compare_bench.cu
    ${MUL_BACKEND_SRC}
    ${SRC_DIR}/ops/carry/carry_norm.cu
    ${SRC_DIR}/ops/shift/shift.cu
    ${SRC_DIR}/ops/sub/sub.cu
    ${SRC_DIR}/mod/batch_mod_ctx.cu
    ${SRC_DIR}/perf/mod_perf.cu
    ${SRC_DIR}/mod/reductions/montgomery.cu
    ${SRC_DIR}/mod/reductions/barrett.cu
)

target_include_directories(bench_mr_gpu PRIVATE
    ${GMP_INCLUDE}
    ${SRC_DIR}
)

target_link_libraries(bench_mr_gpu PRIVATE
    GPUNTT::ntt
    CUDA::cudart
    CUDA::cufft
    ${GMP_LIB}
    stdc++
    m
)
if(MR_USE_GPUFFT)
    target_link_libraries(bench_mr_gpu PRIVATE GPUFFT::fft)
endif()

# Forward GPU-NTT flags from params.cmake to the compiler
if(GPUNTT_CC89)
    target_compile_definitions(bench_mr_gpu PRIVATE CC_89)
endif()
target_compile_definitions(bench_mr_gpu PRIVATE
    GPUNTT_NTT_LAYOUT=${GPUNTT_NTT_LAYOUT}
)

set_target_properties(bench_mr_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
