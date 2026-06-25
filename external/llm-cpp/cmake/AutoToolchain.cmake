# Auto-detect LibTorch prefix, CUDA toolkit, and GPU arch when not explicitly set.
# Mirrors scripts/race/00_env_check.sh and 01_build_cpp.sh so `cmake -B build .`
# configures the same toolchain the race scripts expect.

find_program(OLMO_PYTHON3 NAMES python3 python)
find_program(OLMO_NVIDIA_SMI NAMES nvidia-smi)

# LibTorch via pip-installed torch (before find_package(Torch)).
if(OLMO_PYTHON3)
  execute_process(
    COMMAND ${OLMO_PYTHON3} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _OLMO_LIBTORCH_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _OLMO_LIBTORCH_RC)
  if(_OLMO_LIBTORCH_RC EQUAL 0 AND _OLMO_LIBTORCH_PREFIX AND IS_DIRECTORY "${_OLMO_LIBTORCH_PREFIX}")
    list(FIND CMAKE_PREFIX_PATH "${_OLMO_LIBTORCH_PREFIX}" _OLMO_PREFIX_IDX)
    if(_OLMO_PREFIX_IDX LESS 0)
      list(PREPEND CMAKE_PREFIX_PATH "${_OLMO_LIBTORCH_PREFIX}")
      set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE STRING
        "Semicolon-separated list of package install prefixes" FORCE)
      message(STATUS "LibTorch prefix (from python): ${_OLMO_LIBTORCH_PREFIX}")
    endif()
  endif()
endif()

# Pin nvcc + CUDAToolkit_ROOT to the validated nvcc on PATH.
find_program(OLMO_NVCC NAMES nvcc)
if(OLMO_NVCC)
  if(NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "CMAKE_CUDA_COMPILER-NOTFOUND")
    set(CMAKE_CUDA_COMPILER "${OLMO_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
  if(NOT DEFINED CUDAToolkit_ROOT OR NOT CUDAToolkit_ROOT)
    get_filename_component(_OLMO_NVCC_BIN_DIR "${OLMO_NVCC}" DIRECTORY)
    get_filename_component(_OLMO_CUDA_HOME "${_OLMO_NVCC_BIN_DIR}" DIRECTORY)
    set(CUDAToolkit_ROOT "${_OLMO_CUDA_HOME}" CACHE PATH "CUDA toolkit root" FORCE)
    message(STATUS "CUDA toolkit root (from nvcc): ${CUDAToolkit_ROOT}")
  endif()
endif()

# Target exactly this GPU when OLMO_AUTO_GPU_ARCH=ON (default).
option(OLMO_AUTO_GPU_ARCH "Auto-detect GPU compute capability for CMAKE_CUDA_ARCHITECTURES" ON)
if(OLMO_AUTO_GPU_ARCH)
  set(_OLMO_CC "")
  set(_OLMO_GPU_ARCH_FILE "${CMAKE_SOURCE_DIR}/scripts/race/results/.gpu_arch")
  if(EXISTS "${_OLMO_GPU_ARCH_FILE}")
    file(READ "${_OLMO_GPU_ARCH_FILE}" _OLMO_CC)
    string(REGEX REPLACE "[\r\n].*" "" _OLMO_CC "${_OLMO_CC}")  # first line (heal stale multi-GPU cache)
    string(STRIP "${_OLMO_CC}" _OLMO_CC)
  endif()
  if(NOT _OLMO_CC AND OLMO_NVIDIA_SMI)
    execute_process(
      COMMAND ${OLMO_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _OLMO_CC_DOTTED
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _OLMO_CC_RC)
    if(_OLMO_CC_RC EQUAL 0 AND _OLMO_CC_DOTTED)
      # Multi-GPU boxes print one compute_cap line PER GPU ("9.0\n9.0"); keep
      # only the first (all GPUs share an arch), else CMAKE_CUDA_ARCHITECTURES
      # gets a newline and breaks the math/arch parsing below.
      string(REGEX REPLACE "[\r\n].*" "" _OLMO_CC_DOTTED "${_OLMO_CC_DOTTED}")
      string(REPLACE "." "" _OLMO_CC "${_OLMO_CC_DOTTED}")
      string(STRIP "${_OLMO_CC}" _OLMO_CC)
    endif()
  endif()
  if(_OLMO_CC)
    set(CMAKE_CUDA_ARCHITECTURES "${_OLMO_CC}" CACHE STRING
      "CUDA arch list (auto-detected sm_${_OLMO_CC})" FORCE)
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts/race/results")
    file(WRITE "${_OLMO_GPU_ARCH_FILE}" "${_OLMO_CC}")
    message(STATUS "CUDA architectures (auto): ${CMAKE_CUDA_ARCHITECTURES}")

    # PyTorch ignores CMAKE_CUDA_ARCHITECTURES and compiles for sm_50+ by
    # default, which breaks WMMA headers (nvcuda::wmma needs sm_70+).
    string(LENGTH "${_OLMO_CC}" _OLMO_CC_LEN)
    if(_OLMO_CC_LEN GREATER_EQUAL 3)
      math(EXPR _OLMO_TORCH_MAJOR "${_OLMO_CC} / 10")
      set(_OLMO_TORCH_MINOR "0")
    else()
      math(EXPR _OLMO_TORCH_MAJOR "${_OLMO_CC} / 10")
      math(EXPR _OLMO_TORCH_MINOR "${_OLMO_CC} % 10")
    endif()
    set(TORCH_CUDA_ARCH_LIST "${_OLMO_TORCH_MAJOR}.${_OLMO_TORCH_MINOR}"
      CACHE STRING "CUDA arch list for PyTorch (sm_${_OLMO_CC})" FORCE)
    set(ENV{TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST}")
    message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
  endif()
endif()
