cmake_minimum_required(VERSION 3.18)
project(olmo_cpp LANGUAGES CXX)

# LibTorch prefix, CUDA toolkit, and GPU arch — before CUDA language probe.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AutoToolchain.cmake)

# Default: "release with debug info" → RelWithDebInfo (-O2 + debug symbols on GCC/Clang).
# Single-config generators only (Ninja/Make); Visual Studio/Xcode use --config RelWithDebInfo.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Release with debug info (optimized + symbols). Alternatives: Debug, Release, MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)







# Link-time optimization for better inlining across translation units
# Disabled on Linux/GCC due to cross-library LTO linker issues with vendored deps
option(OLMO_ENABLE_LTO "Enable link-time optimization" OFF)

# Profile-guided optimization (item CC.2). Disabled by default; the
# two-stage workflow is documented at the top of cmake/PGO.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PGO.cmake OPTIONAL)
if(OLMO_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(STATUS "LTO not supported: ${ipo_error}")
    endif()
else()
    message(STATUS "LTO disabled (use -DOLMO_ENABLE_LTO=ON to enable)")
endif()

# Enable CUDA if available (for Jetstream H100, etc.)
include(CheckLanguage)

# Prefer nvcc from CUDAToolkit_ROOT when set (-DCUDAToolkit_ROOT=...), so we don't pick
# /usr/bin/nvcc while a newer CUDA's ptxas appears earlier on PATH (mixed toolchains break).
if(NOT CMAKE_CUDA_COMPILER AND DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

# CMake's CUDA compiler probe defaults to sm_52. CUDA 13+ ptxas rejects sm_52; CUDA 12+
# also needs real archs for BF16 kernels. Set before check_language / enable_language.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # 120 = Blackwell (RTX 50-series, e.g. 5060 Ti). REQUIRES CUDA Toolkit >= 12.8;
  # older nvcc rejects sm_120. If the build fails on "compute_120" with an older
  # toolkit, set -DCMAKE_CUDA_ARCHITECTURES=89 (Ada PTX is forward-compatible and
  # JITs onto Blackwell on first launch — slower first step, but it runs).
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120" CACHE STRING
    "CUDA arch list (86=RTX 3060, 89=RTX 40xx, 90=H100, 120=RTX 50xx/Blackwell). Must be set before first CUDA probe.")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_compile_definitions(USE_CUDA=1)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (version ${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    include(FindCUDAToolkit)
    if(CUDAToolkit_VERSION)
      message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
    endif()
endif()

# Find LibTorch (CPU or CUDA build)
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# Detect which autocast C++ API is available (changed across PyTorch versions).
# Use STATIC_LIBRARY target type so try_compile only compiles (no link step),
# avoiding protobuf and other transitive dependency issues.
include(CheckCXXSourceCompiles)
set(_SAVE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_INCLUDES ${TORCH_INCLUDE_DIRS})
set(CMAKE_REQUIRED_FLAGS "${TORCH_CXX_FLAGS} -std=c++17")

# Probe 1: newest API — is_autocast_enabled(at::kCUDA) (PyTorch >= 2.4)
check_cxx_source_compiles("
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
void test() { (void)at::autocast::is_autocast_enabled(at::kCUDA); }
" HAVE_AUTOCAST_DEVICE_API)

# Probe 2: mid-era — is_autocast_gpu_enabled() (some PyTorch 2.x builds)
if(NOT HAVE_AUTOCAST_DEVICE_API)
    check_cxx_source_compiles("
    #include <ATen/autocast_mode.h>
    void test() { (void)at::autocast::is_autocast_gpu_enabled(); }
    " HAVE_AUTOCAST_GPU_API)
endif()

# Probe 3: older PyTorch 2.1 — is_enabled() + set_enabled(bool) (no device param, GPU-only)
if(NOT HAVE_AUTOCAST_DEVICE_API AND NOT HAVE_AUTOCAST_GPU_API)
    check_cxx_source_compiles("
    #include <ATen/autocast_mode.h>
    void test() { at::autocast::set_enabled(true); (void)at::autocast::is_enabled(); }
    " HAVE_AUTOCAST_LEGACY_API)
endif()

if(HAVE_AUTOCAST_DEVICE_API)
    add_compile_definitions(OLMO_AUTOCAST_DEVICE_API=1)
    message(STATUS "Autocast API: device-parameterized (PyTorch >= 2.4)")
elseif(HAVE_AUTOCAST_GPU_API)
    add_compile_definitions(OLMO_AUTOCAST_GPU_API=1)
    message(STATUS "Autocast API: GPU-specific (PyTorch 2.x)")
elseif(HAVE_AUTOCAST_LEGACY_API)
    add_compile_definitions(OLMO_AUTOCAST_LEGACY_API=1)
    message(STATUS "Autocast API: legacy is_enabled/set_enabled (PyTorch 2.0-2.1)")
else()
    message(STATUS "Autocast API: dispatch-key fallback (autocast may not work)")
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_SAVE_TARGET_TYPE})
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_FLAGS)

# cnpy for .npy token files (compiled directly into olmo_cpp to avoid GCC link-order issues)
# add_subdirectory(third_party/cnpy)

# Optional: nlohmann_json for config and prepare_data (BPE tokenizer)
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS /usr/include /usr/local/include /opt/homebrew/include
          $ENV{HOME}/.local/include
)
if(NOT JSON_INCLUDE_DIR)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
    get_target_property(JSON_INCLUDE_DIR nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    list(GET JSON_INCLUDE_DIR 0 JSON_INCLUDE_DIR)
endif()
if(JSON_INCLUDE_DIR)
    add_compile_definitions(HAS_NLOHMANN_JSON=1)
endif()

# DDP requires Gloo (not in pip LibTorch). Check for gloo.
find_path(GLOO_INCLUDE_DIR gloo/algorithm.h HINTS /usr/local/include /opt/homebrew/include)
option(OLMO_USE_DDP "Build with DDP (requires Gloo)" OFF)
# NCCL DDP: GPU collectives via c10d::ProcessGroupNCCL (ships in pip torch as
# nvidia-nccl-cu12 — no Gloo headers needed). Preferred for H100 multi-GPU.
option(OLMO_USE_NCCL "Build with NCCL DDP (no Gloo needed)" OFF)

# Main library: model components + data + training infrastructure
set(OLMO_SOURCES
    # cnpy (vendored .npy loader)
    third_party/cnpy/cnpy.cpp
    # Config
    src/config.cpp
    # Data pipeline
    src/data/token_dataset.cpp
    src/data/bpe_tokenizer.cpp
    src/data/collator.cpp
    src/data/source_mixture.cpp
    src/data/vsl_dataset.cpp
    src/data/structural_tokenizer.cpp
    src/data/composable/data_loader.cpp
    src/data/composable/document_source.cpp
    src/data/composable/instance_source.cpp
    src/data/composable/token_source.cpp
    # Model: core
    src/model/layer_norm.cpp
    src/model/layer_norm_variants.cpp
    src/model/rope.cpp
    src/model/rope_scaling.cpp
    src/model/feed_forward.cpp
    src/model/attention.cpp
    src/model/attention_config.cpp
    src/model/block.cpp
    src/model/block_variants.cpp
    src/model/lm_head.cpp
    src/model/mtp_head.cpp
    src/model/transformer.cpp
    src/model/fused_attention.cpp
    src/model/fused_block.cpp
    src/model/fused_transformer.cpp
    src/model/mup_init.cpp
    src/model/convolution.cpp
    src/model/gated_attention.cpp
    # Model: MoE
    src/model/moe/router.cpp
    src/model/moe/loss.cpp
    src/model/moe/mlp.cpp
    src/model/moe/moe.cpp
    # Optimizers
    src/optim/lion.cpp
    src/optim/muon.cpp
    src/optim/dion.cpp
    src/optim/foreach_adamw.cpp
    src/optim/grad_clip.cpp
    src/optim/skip_step.cpp
    src/optim/sgp.cpp
    src/optim/sgp_v2.cpp
    src/optim/scheduler.cpp
    # Evaluation
    src/eval/metrics.cpp
    src/eval/evaluator.cpp
    src/eval/lm_evaluator.cpp
    # Float8
    src/float8/float8.cpp
    # Training infrastructure
    src/train/checkpoint.cpp
    src/train/activation_checkpoint.cpp
    src/train/grad_scaler.cpp
    src/train/callbacks/all_callbacks.cpp
    # IO abstraction
    src/io/filesystem.cpp
    # HuggingFace conversion
    src/nn/hf_convert.cpp
    # Multi-Resolution Embedding (DC-MRE)
    src/nn/multi_res_embedding.cpp
    # Compute backend
    src/backend/backend.cpp
    src/backend/simd_backend.cpp
    src/backend/arena.cpp
    src/backend/fused_ops.cpp
    src/backend/kernels/simd_elementwise.cpp
    src/backend/cuda_backend.cpp
    src/backend/topp_radix.cpp  # fast-inference [7]
    src/backend/fused_lm_head_sample.cpp  # fast-inference [6]
    src/backend/gpu_sample.cpp  # device-resident sampler (every decode path)
    src/backend/paged_attention.cpp  # fast-inference [9] (CPU ref + dispatch)
    src/backend/lm_head_gemv.cpp  # fast-inference [11]
    src/backend/flash_decode.cpp  # fast-inference [12]
    src/backend/sparse_attn.cpp  # SubQ Phase-1: NSA-style content-selected attention
    src/backend/persistent_decode.cpp  # fast-inference [13] (stub)
    src/nn/quant.cpp  # fast-inference [14] FP8 + [15] INT4
    src/generate/draft_model_speculative.cpp  # fast-inference [17]
    src/model/paged_kv_cache.cpp  # fast-inference [1] partial (concat shim)
    src/model/block_manager.cpp  # fast-inference [9] (page allocator)
    src/model/shared_block_pool.cpp  # fast-inference [I-7] (ref-counted shared pool)
    src/serve/scheduler.cpp  # fast-inference [I-2] (multi-request scheduler)
    src/train/async_loss_reader.cpp  # AA (non-blocking loss readout)
    src/backend/topk_argmax.cpp  # K (fused GEMV+argmax dispatcher)
    src/backend/fused_qkv_rope.cpp  # G (fused QKV + reshape + RoPE dispatcher)
    src/backend/fused_ffn.cpp       # I (fused SwiGLU FFN dispatcher)
    src/backend/cublas_direct.cpp   # L (direct cuBLASLt fast path)
    src/optim/eight_bit_adamw.cpp   # O (8-bit Adam moment states)
    src/backend/int4_kv.cpp         # U (INT4 KV codec)
    # NOTE: tma_loads.cpp (TMA descriptor builder) is compiled into olmo_kernels,
    # not here — its only caller is fused_ffn_tma.cu. See kernels/CMakeLists.txt.
    src/backend/fp8_cublaslt.cpp    # S (FP8 cuBLASLt entry point)
    src/generate/tree_speculative.cpp  # 8.1 (tree speculative)
    src/backend/fused_qkv_rope_autograd.cpp  # G backward (autograd-aware)
    src/backend/fused_ffn_autograd.cpp       # I backward (autograd-aware)
    src/backend/fused_lm_head_ce.cpp         # A3 — fused LM-head + CE dispatcher
    src/backend/fused_lm_head_ce_autograd.cpp  # A3 backward (recompute + grad routing)
    # cuda_ops_autograd.cpp moved to kernels/olmo_ops_autograd.cpp (shared lib) — linker strips
    # static-lib objects with no exported symbols; see kernels/olmo_ops_autograd.cpp for details.
    src/backend/subq_backward.cpp            # HH SubQ STE backward
    # Seed and profiling infrastructure
    src/seed.cpp
    src/profiler.cpp
    # Speculative decoding
    src/generate/speculative_decode.cpp
)
# Advanced parallelism strategies (PP / FSDP / CP / EP) are wired up at the
# class/header level but never instantiated by main.cpp on this branch. They
# carry no-op send/recv stubs that aren't safe to ship as "real" — pretending
# they work hides bugs. Default OFF so a stock build doesn't compile the
# stubs; flip ON when you actually wire one of them into the trainer and
# replace its TODO body with a real c10d call. The standard DDP path (data
# parallel + tensor parallel via NCCL allreduce) stays under OLMO_USE_DDP.
option(OLMO_USE_ADVANCED_PARALLEL
       "Build pipeline / FSDP / context / expert parallel (off; stubs only)" OFF)

# zero1 is always compiled: with no backend it falls through to a no-op
# pass-through partition + allgather, so call sites work in single-rank
# mode without conditional compilation.
list(APPEND OLMO_SOURCES src/distributed/zero1.cpp)

if(OLMO_USE_NCCL)
    # NCCL multi-GPU path: ddp.cpp (DDP) + fsdp.cpp (FSDP/ZeRO-3 sharding),
    # both with the ProcessGroupNCCL backend. No Gloo needed.
    list(APPEND OLMO_SOURCES src/distributed/ddp.cpp src/distributed/fsdp.cpp)
    message(STATUS "Distributed: NCCL DDP + FSDP ENABLED (olmo_train multi-GPU)")
elseif(OLMO_USE_DDP AND GLOO_INCLUDE_DIR)
    list(APPEND OLMO_SOURCES
        src/distributed/ddp.cpp
        src/distributed/tensor_parallel.cpp
    )
    if(OLMO_USE_ADVANCED_PARALLEL)
        list(APPEND OLMO_SOURCES
            src/distributed/pipeline_parallel.cpp
            src/distributed/fsdp.cpp
            src/distributed/context_parallel.cpp
            src/distributed/expert_parallel.cpp
        )
        message(STATUS "Advanced parallelism (PP/FSDP/CP/EP): ENABLED")
    else()
        message(STATUS "Advanced parallelism (PP/FSDP/CP/EP): DISABLED "
                       "(set -DOLMO_USE_ADVANCED_PARALLEL=ON to compile stubs)")
    endif()
else()
    list(APPEND OLMO_SOURCES src/distributed/ddp_stub.cpp)
endif()
add_library(olmo_cpp STATIC ${OLMO_SOURCES})
if(OLMO_USE_NCCL)
    # PUBLIC so the macro propagates to targets that link olmo_cpp — notably
    # olmo_train, which compiles src/train.cpp (the FSDP/DDP call sites) into
    # its OWN object, not olmo_cpp. PRIVATE here left train.cpp's FSDP wiring
    # compiled out -> "fsdp=1 but built without NCCL" at runtime.
    target_compile_definitions(olmo_cpp PUBLIC OLMO_HAS_NCCL=1)
elseif(OLMO_USE_DDP AND GLOO_INCLUDE_DIR)
    target_compile_definitions(olmo_cpp PUBLIC OLMO_HAS_DDP=1)
endif()
target_include_directories(olmo_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
    ${TORCH_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIR}
)
target_link_libraries(olmo_cpp PUBLIC ${TORCH_LIBRARIES} ZLIB::ZLIB)
if(CUDA_FOUND OR CMAKE_CUDA_COMPILER)
  target_link_libraries(olmo_cpp PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cudart)
endif()
target_compile_options(olmo_cpp PRIVATE -O3 -march=native)

# ========================================================================
# Zero-Wait Trainer (zwt) — LibTorch-free training framework
# Own tensor/allocator/ops/layers/optim. Built as an independent static lib
# alongside olmo_cpp; zwt has zero dependency on torch. Keeps cuBLAS for
# GEMMs (industry ceiling) and dispatches elementwise / norm / xent / attn
# to custom bf16 CUDA kernels.
# ========================================================================
set(ZWT_CPP_SOURCES
    zwt/src/core/allocator.cpp
    zwt/src/core/determinism.cpp
    zwt/src/core/graph.cpp
    zwt/src/core/profiler.cpp
    zwt/src/core/tensor.cpp
    zwt/src/core/stream.cpp
    zwt/src/dist/ddp.cpp
    zwt/src/dist/comm.cpp
    zwt/src/dist/tensor_parallel.cpp
    zwt/src/ops/gemm.cpp
    zwt/src/ops/elementwise.cpp
    zwt/src/ops/norm.cpp
    zwt/src/ops/xent.cpp
    zwt/src/ops/attn.cpp
    zwt/src/ops/flash_attn.cpp
    zwt/src/ops/gemm_wgmma_stub.cpp
    zwt/src/ops/rope.cpp
    zwt/src/layers/module.cpp
    zwt/src/layers/parameter.cpp
    zwt/src/layers/linear.cpp
    zwt/src/layers/rmsnorm.cpp
    zwt/src/layers/embedding.cpp
    zwt/src/layers/ffn.cpp
    zwt/src/layers/attention.cpp
    zwt/src/layers/transformer_block.cpp
    zwt/src/layers/transformer.cpp
    zwt/src/optim/adamw.cpp
    zwt/src/optim/grad_clip.cpp
    zwt/src/data/token_loader.cpp
    zwt/src/train/checkpoint.cpp
    zwt/src/train/config.cpp
)
set(ZWT_CUDA_SOURCES
    zwt/src/ops/elementwise.cu
    zwt/src/ops/norm.cu
    zwt/src/ops/xent.cu
    zwt/src/ops/attn.cu
    zwt/src/ops/rope.cu
    zwt/src/optim/adamw.cu
    zwt/src/optim/grad_clip.cu
    zwt/src/layers/embedding.cu
)

# NCCL: needed only for real multi-GPU DDP.
#
# Modern PyTorch wheels (cu12+) DON'T ship nccl directly under
# torch/{lib,include}; instead they pull `nvidia-nccl-cu12` as a pip
# sub-dep and that lands at  <site-packages>/nvidia/nccl/{lib,include}/.
# So we ask Python where that package lives and add it to the search
# hints. Falls back to the classic system locations (/usr/local/cuda,
# Debian's libnccl-dev) when the pip layout isn't in use.
#
# If we can't find both header + library, world_size>1 errors at runtime
# with a clear message — the rest of the build is unaffected.
if(CMAKE_CUDA_COMPILER)
    # Probe Python for the pip-installed nvidia-nccl-cu12 location. Quiet
    # on failure (the fallback HINTS still cover system installs).
    set(_PIP_NCCL_DIR "")
    find_program(PYTHON3_EXE NAMES python3 python)
    if(PYTHON3_EXE)
        execute_process(
            COMMAND ${PYTHON3_EXE} -c
                "import os; import nvidia.nccl as n; print(os.path.dirname(n.__file__))"
            OUTPUT_VARIABLE _PIP_NCCL_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _PIP_NCCL_RC)
        if(_PIP_NCCL_RC EQUAL 0 AND _PIP_NCCL_DIR)
            message(STATUS "NCCL pip package: ${_PIP_NCCL_DIR}")
        endif()
    endif()

    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS ${_PIP_NCCL_DIR}/include
              ${TORCH_INSTALL_PREFIX}/include
              /usr/local/cuda/include
              /usr/include)
    find_library(NCCL_LIBRARY
        NAMES nccl libnccl.so.2 libnccl.so
        HINTS ${_PIP_NCCL_DIR}/lib
              ${TORCH_INSTALL_PREFIX}/lib
              /usr/local/cuda/lib64
              /usr/lib/x86_64-linux-gnu)
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        message(STATUS "NCCL: ${NCCL_LIBRARY} (headers: ${NCCL_INCLUDE_DIR})")
        list(APPEND ZWT_CUDA_SOURCES zwt/src/dist/nccl_backend.cu)
        set(ZWT_HAVE_NCCL TRUE)
        # olmo_cpp's ddp.cpp (OLMO_USE_NCCL) pulls in ProcessGroupNCCL.hpp ->
        # NCCLUtils.hpp -> <nccl.h>; pip torch doesn't expose that header on the
        # default include path, so give the target the pip NCCL header + lib.
        if(OLMO_USE_NCCL)
            target_include_directories(olmo_cpp PRIVATE ${NCCL_INCLUDE_DIR})
            target_link_libraries(olmo_cpp PUBLIC ${NCCL_LIBRARY})
            message(STATUS "  -> wired NCCL headers/lib into olmo_cpp (multi-GPU DDP)")
        endif()
    else()
        message(STATUS "NCCL: not found — multi-GPU DDP will be disabled at runtime.")
        message(STATUS "  hints tried: pip=${_PIP_NCCL_DIR}, torch=${TORCH_INSTALL_PREFIX}, /usr/local/cuda, /usr/{include,lib}")
        message(STATUS "  to fix: pip install nvidia-nccl-cu12   (or apt install libnccl-dev)")
        set(ZWT_HAVE_NCCL FALSE)
    endif()
else()
    set(ZWT_HAVE_NCCL FALSE)
endif()

if(CMAKE_CUDA_COMPILER)
    add_library(zwt STATIC ${ZWT_CPP_SOURCES} ${ZWT_CUDA_SOURCES})
    set_target_properties(zwt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(zwt PUBLIC CUDA::cublas CUDA::cudart)
    if(ZWT_HAVE_NCCL)
        target_include_directories(zwt PRIVATE ${NCCL_INCLUDE_DIR})
        target_link_libraries(zwt PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(zwt PUBLIC ZWT_HAVE_NCCL=1)
    endif()
else()
    add_library(zwt STATIC ${ZWT_CPP_SOURCES})
endif()
target_include_directories(zwt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/zwt/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(zwt PRIVATE -O3 -march=native -Wall)
if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(zwt PUBLIC USE_CUDA=1)
endif()

# --------------------------------------------------------------------------
# Hopper WGMMA GEMM via CUTLASS 3.x — opt-in for sm_90 hosts.
#
# We build this as a *separate* static lib (zwt_wgmma) so we can target
# sm_90a in isolation — WGMMA instructions are only emitted under the
# `a` arch suffix, and the main `zwt` lib stays portable across
# sm_80..sm_90. CUTLASS is fetched on demand; without the flag it is
# never downloaded and the stub TU in zwt/src/ops/gemm_wgmma_stub.cpp
# provides the symbols.
# --------------------------------------------------------------------------
option(ZWT_USE_WGMMA "Build Hopper WGMMA GEMM via CUTLASS 3.x" OFF)
if(ZWT_USE_WGMMA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "ZWT_USE_WGMMA requires CUDA.")
    endif()

    include(FetchContent)
    FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG        v3.5.1
        GIT_SHALLOW    TRUE
    )
    # We only need the headers — do not invoke CUTLASS's CMake (it builds
    # many profilers / tests we don't want).
    FetchContent_GetProperties(cutlass)
    if(NOT cutlass_POPULATED)
        FetchContent_Populate(cutlass)
    endif()

    add_library(zwt_wgmma STATIC zwt/src/ops/gemm_wgmma.cu)
    # Plain "90" compiles for sm_90 (no WGMMA). We need sm_90a. Disable
    # the CMake-managed arch list for this target and emit the gencode
    # flag by hand.
    set_target_properties(zwt_wgmma PROPERTIES
        CUDA_ARCHITECTURES OFF
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(zwt_wgmma PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_include_directories(zwt_wgmma PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/zwt/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include)
    target_compile_definitions(zwt_wgmma PRIVATE
        ZWT_USE_WGMMA=1 USE_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(zwt_wgmma PRIVATE CUDA::cudart)

    # Tell the main library it can call into gemm_wgmma() unconditionally.
    # (The stub TU self-disables when the macro is set so we don't get
    # duplicate symbols.)
    target_compile_definitions(zwt PRIVATE ZWT_USE_WGMMA=1)
    target_link_libraries(zwt PUBLIC zwt_wgmma)
    message(STATUS "ZWT_USE_WGMMA: Hopper WGMMA path enabled (sm_90a, "
                   "CUTLASS @ ${cutlass_SOURCE_DIR})")
endif()

# ZWT end-to-end training demo binary — exercises every path of the framework.
add_executable(zwt_train zwt/tools/zwt_train.cpp)
target_link_libraries(zwt_train PRIVATE zwt)
target_compile_options(zwt_train PRIVATE -O3 -march=native)

add_executable(zwt_pretrain zwt/tools/zwt_pretrain.cpp)
target_link_libraries(zwt_pretrain PRIVATE zwt)
target_compile_options(zwt_pretrain PRIVATE -O3 -march=native)

add_executable(zwt_export_hf zwt/tools/export_hf.cpp)
target_link_libraries(zwt_export_hf PRIVATE zwt)
target_compile_options(zwt_export_hf PRIVATE -O3 -march=native)

add_executable(zwt_numerical_audit zwt/tools/numerical_audit.cpp)
target_link_libraries(zwt_numerical_audit PRIVATE zwt)
target_compile_options(zwt_numerical_audit PRIVATE -O3 -march=native)

add_executable(zwt_kernel_tests zwt/tests/kernel_tests.cpp)
target_link_libraries(zwt_kernel_tests PRIVATE zwt)
target_compile_options(zwt_kernel_tests PRIVATE -O2)

add_executable(zwt_ddp_bucket_tests zwt/tests/ddp_bucket_tests.cpp)
target_link_libraries(zwt_ddp_bucket_tests PRIVATE zwt)
target_compile_options(zwt_ddp_bucket_tests PRIVATE -O2)

add_executable(zwt_ddp_loopback_tests zwt/tests/ddp_loopback_tests.cpp)
target_link_libraries(zwt_ddp_loopback_tests PRIVATE zwt)
target_compile_options(zwt_ddp_loopback_tests PRIVATE -O2)

add_executable(zwt_tp_tests zwt/tests/tp_tests.cpp)
target_link_libraries(zwt_tp_tests PRIVATE zwt)
target_compile_options(zwt_tp_tests PRIVATE -O2)

add_executable(zwt_flash_attn_tests zwt/tests/flash_attn_tests.cpp)
target_link_libraries(zwt_flash_attn_tests PRIVATE zwt)
target_compile_options(zwt_flash_attn_tests PRIVATE -O2)

# Hopper WGMMA bench + numerical test — only buildable on CUDA hosts, and
# really meaningful only with -DZWT_USE_WGMMA=ON on sm_90.
if(CMAKE_CUDA_COMPILER)
    add_executable(zwt_wgmma_bench zwt/tools/wgmma_bench.cpp)
    target_link_libraries(zwt_wgmma_bench PRIVATE zwt)
    target_compile_options(zwt_wgmma_bench PRIVATE -O3 -march=native)

    add_executable(zwt_wgmma_tests zwt/tests/wgmma_tests.cpp)
    target_link_libraries(zwt_wgmma_tests PRIVATE zwt)
    target_compile_options(zwt_wgmma_tests PRIVATE -O2)
endif()

add_executable(zwt_graph_bench zwt/tools/graph_bench.cpp)
target_link_libraries(zwt_graph_bench PRIVATE zwt)
target_compile_options(zwt_graph_bench PRIVATE -O3 -march=native)

# CUDA kernels (auto-enabled when CUDA compiler is found)
option(OLMO_BUILD_KERNELS "Build custom CUDA kernels" ON)
if(CMAKE_CUDA_COMPILER AND OLMO_BUILD_KERNELS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/CMakeLists.txt")
    add_subdirectory(kernels)
    target_link_libraries(olmo_cpp PUBLIC olmo_kernels)
    target_compile_definitions(olmo_cpp PRIVATE OLMO_HAS_CUDA_KERNELS=1)
    message(STATUS "Building custom CUDA kernels")
endif()

# Training executable
add_executable(olmo_train
    src/main.cpp
    src/train.cpp
)
target_link_libraries(olmo_train PRIVATE olmo_cpp)
# if(CMAKE_CUDA_COMPILER AND OLMO_BUILD_KERNELS)
#     target_link_libraries(olmo_train PRIVATE olmo_kernels)
# endif()
target_compile_options(olmo_train PRIVATE -O3 -march=native)

# Checkpoint converter tool
add_executable(convert_checkpoint tools/convert_checkpoint.cpp)
target_link_libraries(convert_checkpoint PRIVATE olmo_cpp)
target_compile_options(convert_checkpoint PRIVATE -O3)

# Interactive chat CLI
add_executable(chat tools/chat.cpp)
target_link_libraries(chat PRIVATE olmo_cpp)
target_include_directories(chat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(chat PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(chat PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Inference micro-benchmark (fast-inference roadmap item [3]).
# Greedy-only deterministic decode with JSON timing output.
add_executable(bench_chat tools/bench_chat.cpp)
target_link_libraries(bench_chat PRIVATE olmo_cpp)
target_include_directories(bench_chat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(bench_chat PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(bench_chat PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Attention-kernel-isolated benchmark — used for the SubQ head-to-head.
# Runs only the SDPA op (no model load) across a sweep of seq lens and modes
# {dense, fa2, fa3, sparse} for stages {prefill, decode}. Emits CSV.
add_executable(bench_attn tools/bench_attn.cpp)
target_link_libraries(bench_attn PRIVATE olmo_cpp)
target_include_directories(bench_attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bench_attn PRIVATE -O3 -march=native)

# Bucket-radix top-p kernel unit test (fast-inference [7]).
# CPU-only; verifies the algorithm against a sort-based reference.
add_executable(test_topp_radix tools/test_topp_radix.cpp)
target_link_libraries(test_topp_radix PRIVATE olmo_cpp)
target_include_directories(test_topp_radix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_topp_radix PRIVATE -O3 -march=native)

# PagedKVCache equivalence test (fast-inference [1]).
# CPU-only; checks that the paged storage returns the same K/V views as the
# legacy concat-based LayerKVCache across a mix of prefill/decode appends and
# a snapshot+rollback sequence.
add_executable(test_paged_kv tools/test_paged_kv.cpp)
target_link_libraries(test_paged_kv PRIVATE olmo_cpp)
target_include_directories(test_paged_kv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_paged_kv PRIVATE -O3 -march=native)

# INT4 post-training weight-only quantizer (item T).
add_executable(quantize_int4 tools/quantize_int4.cpp)
target_link_libraries(quantize_int4 PRIVATE olmo_cpp)
target_include_directories(quantize_int4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(quantize_int4 PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(quantize_int4 PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# SharedBlockPool unit test (fast-inference [I-7]).
# CPU-only; verifies alloc/dec_ref, content-hash lookup, hash chaining,
# and cross-request prefix sharing all hold.
add_executable(test_prefix_cache tools/test_prefix_cache.cpp)
target_link_libraries(test_prefix_cache PRIVATE olmo_cpp)
target_include_directories(test_prefix_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_prefix_cache PRIVATE -O3 -march=native)

# Scheduler integration test (fast-inference [I-2]).
# Loads the 125M model + tokenizer; admits 2 requests with a shared
# prefix; verifies prefix-page sharing and per-request completion.
add_executable(test_scheduler tools/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE olmo_cpp)
target_include_directories(test_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(test_scheduler PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(test_scheduler PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# A3 — fused LM-head + softmax-CE parity vs torch::nn::functional::cross_entropy.
add_executable(test_fused_ce tools/test_fused_ce.cpp)
target_link_libraries(test_fused_ce PRIVATE olmo_cpp)
target_include_directories(test_fused_ce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_fused_ce PRIVATE -O3 -march=native)

# A2 — fused QKV+RoPE forward/backward parity vs the discrete ATen chain.
add_executable(test_fused_qkv_rope tools/test_fused_qkv_rope.cpp)
target_link_libraries(test_fused_qkv_rope PRIVATE olmo_cpp)
target_include_directories(test_fused_qkv_rope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_fused_qkv_rope PRIVATE -O3 -march=native)

# CUDA-only parity: exercises the WMMA UB fix, AutogradCUDA wrappers,
# and fused-LM-head-CE / QKV+RoPE / FFN on real hardware. Compiles
# everywhere; main() bails with exit code 2 if CUDA isn't available.
add_executable(test_cuda_parity tools/test_cuda_parity.cpp)
target_link_libraries(test_cuda_parity PRIVATE olmo_cpp)
target_include_directories(test_cuda_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_cuda_parity PRIVATE -O3 -march=native)

# Fused LM-head + Gumbel-max sampler unit test (fast-inference [6]).
# CPU-only; verifies determinism, distribution match vs softmax, and
# temperature-sharpening behavior.
add_executable(test_fused_lm_head tools/test_fused_lm_head.cpp)
target_link_libraries(test_fused_lm_head PRIVATE olmo_cpp)
target_include_directories(test_fused_lm_head PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(test_fused_lm_head PRIVATE -O3 -march=native)

# Param dumper (debug utility)
add_executable(dump_params tools/dump_params.cpp)
target_link_libraries(dump_params PRIVATE olmo_cpp)

# Embedding dumper (post-training inspection: writes .npy + summary).
# Links olmo_cpp for the Transformer / FusedTransformer modules and cnpy
# (compiled into olmo_cpp) for the .npy writer. nlohmann/json is optional;
# without it the --vocab flag is a no-op.
add_executable(dump_embeddings tools/dump_embeddings.cpp)
target_include_directories(dump_embeddings PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
  ${JSON_INCLUDE_DIR}
)
target_link_libraries(dump_embeddings PRIVATE olmo_cpp)
target_compile_options(dump_embeddings PRIVATE -O3)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(dump_embeddings PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# HF checkpoint converter
add_executable(convert_hf tools/convert_hf.cpp)
target_link_libraries(convert_hf PRIVATE olmo_cpp)
target_compile_options(convert_hf PRIVATE -O3)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(convert_hf PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Data preparation tool (no LibTorch - standalone, all C++)
add_executable(prepare_data
  tools/prepare_data.cpp
  src/data/simple_tokenizer.cpp
  src/data/bpe_tokenizer.cpp
  src/data/structural_tokenizer.cpp
  third_party/cnpy/cnpy.cpp
)
target_include_directories(prepare_data PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
  ${JSON_INCLUDE_DIR}
)
target_link_libraries(prepare_data PRIVATE ZLIB::ZLIB)
target_compile_options(prepare_data PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(prepare_data PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Token inspection tool (no LibTorch - standalone, all C++)
add_executable(inspect_tokens
  tools/inspect_tokens.cpp
  src/data/bpe_tokenizer.cpp
)
target_include_directories(inspect_tokens PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSON_INCLUDE_DIR}
)
target_compile_options(inspect_tokens PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(inspect_tokens PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Pattern mining tool (no LibTorch - standalone, all C++)
add_executable(mine_patterns
  tools/mine_patterns.cpp
)
target_include_directories(mine_patterns PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(mine_patterns PRIVATE -O3 -march=native)

# Tokenizer benchmark (no LibTorch - standalone, all C++)
add_executable(benchmark_tokenizer
  tools/benchmark_tokenizer.cpp
  src/data/bpe_tokenizer.cpp
  src/data/structural_tokenizer.cpp
)
target_include_directories(benchmark_tokenizer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${JSON_INCLUDE_DIR}
)
target_compile_options(benchmark_tokenizer PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(benchmark_tokenizer PRIVATE HAS_NLOHMANN_JSON=1)
endif()

# Race pipeline (scripts/race/run_all.sh) — incremental custom targets.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Race.cmake)
