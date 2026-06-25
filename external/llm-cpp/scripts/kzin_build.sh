#!/usr/bin/env bash
# ============================================================================
# scripts/kzin_build.sh
#
# Minimal CMake-configure wrapper for the kzin Linux box. Its sole job is to
# pin CMake to a *single* CUDA toolkit (12.1 by default) so that we don't end
# up with a mismatched mix — for example, nvcc from /usr/bin (CUDA 13) and
# headers/libs from /usr/local/cuda-12.1, which produces opaque ptxas/host-
# compiler errors. Pass any extra CMake options as positional arguments.
#
# This script ONLY configures; it does not build. Run `cmake --build build`
# (or `make -C build -j$(nproc)`) afterwards. For a fully managed build,
# prefer scripts/build.sh.
#
# Usage:
#   ./scripts/kzin_build.sh                                  # default config
#   ./scripts/kzin_build.sh -DCMAKE_BUILD_TYPE=Debug         # debug build
#   ./scripts/kzin_build.sh -DOLMO_USE_DDP=ON                # enable DDP
#   CUDA_HOME=/usr/local/cuda-12.4 ./scripts/kzin_build.sh   # override toolkit
#
# --- Reads ---
#   $CUDA_HOME           (env, optional; defaults to /usr/local/cuda-12.1)
#   ./CMakeLists.txt     (project source)
#   /opt/libtorch        (LibTorch install on the kzin machine)
#
# --- Writes / Side effects ---
#   ./build/             (CMake build directory: CMakeCache.txt, generated
#                        build system, etc.)
#
# --- Calls ---
#   cmake (configure-only; the `exec` replaces this shell with cmake)
#
# --- Role in workflow ---
#   One-shot configure step. Run once after `git clone`; rerun only when
#   you change CMake options or want to wipe build/ and start fresh.
# ============================================================================

# Pick the CUDA toolkit to drive the build. The kzin box has CUDA 13's ptxas
# on PATH alongside CUDA 12.1, which CMake will happily mix unless we point
# every CUDA-related variable at one tree.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"

# Replace this shell with cmake so the exit status of `cmake` is our exit
# status (no extra subprocess in pipelines / CI).
#   -S .                              Source tree is the current directory.
#   -B build                          Out-of-source build directory.
#   -DCMAKE_PREFIX_PATH=/opt/libtorch Tell CMake where LibTorch's
#                                     TorchConfig.cmake lives.
#   -DCUDAToolkit_ROOT                Modern (CMake 3.17+) variable used by
#     find_package(CUDAToolkit) for headers and stub libs.
#   -DCUDA_TOOLKIT_ROOT_DIR           Legacy variable still consulted by
#                                     some FindCUDA paths and dependencies.
#   -DCMAKE_CUDA_COMPILER=.../nvcc    Force CMake to use *this* nvcc, not
#                                     whatever nvcc happens to be on PATH.
#   "$@"                              Forward any extra cmake flags from
#                                     the caller (e.g. -DCMAKE_BUILD_TYPE=…).
exec cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  "$@"
