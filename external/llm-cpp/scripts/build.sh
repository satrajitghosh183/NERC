#!/bin/bash
# Build script for llm-cpp on any platform (CPU, MPS, CUDA/H100)
#
# Usage:
#   ./scripts/build.sh                # Auto-detect platform
#   ./scripts/build.sh --cuda         # Force CUDA build
#   ./scripts/build.sh --cpu          # CPU-only build
#   ./scripts/build.sh --clean        # Clean rebuild
#
# Jetstream H100:
#   module load cuda/12.x  # if using module system
#   ./scripts/build.sh --cuda
#
# Dependencies:
#   - CMake 3.18+
#   - C++17 compiler (GCC 8+, Clang 10+)
#   - LibTorch (auto-detected from pip torch or CMAKE_PREFIX_PATH)
#   - ZLIB (usually system-installed)
#   - CUDA Toolkit 12.0+ (for H100)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build"
BUILD_TYPE="Release"
FORCE_CUDA=""
FORCE_CPU=""
CLEAN=""
EXTRA_CMAKE_ARGS=""

for arg in "$@"; do
  case $arg in
    --cuda)   FORCE_CUDA=1 ;;
    --cpu)    FORCE_CPU=1 ;;
    --clean)  CLEAN=1 ;;
    --debug)  BUILD_TYPE="Debug" ;;
    --ddp)    EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DOLMO_USE_DDP=ON" ;;
    --nccl)   EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DOLMO_USE_NCCL=ON" ;;
    --wgmma)  EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DZWT_USE_WGMMA=ON" ;;
    --help|-h)
      echo "Usage: $0 [--cuda|--cpu|--clean|--debug|--ddp|--nccl|--wgmma]"
      echo ""
      echo "  --cuda    Force CUDA build (H100 etc.)"
      echo "  --cpu     CPU-only build"
      echo "  --clean   Wipe build/ before configuring"
      echo "  --debug   -DCMAKE_BUILD_TYPE=Debug"
      echo "  --ddp     Build legacy olmo_cpp DDP path (Gloo)"
      echo "  --nccl    Build multi-GPU olmo_cpp DDP path (NCCL, no Gloo) [WIP]"
      echo "  --wgmma   Enable Hopper WGMMA GEMM in zwt (sm_90a, requires CUDA 12+)"
      exit 0
      ;;
  esac
done

# Clean build if requested
if [ -n "$CLEAN" ] && [ -d "$BUILD_DIR" ]; then
  echo "Cleaning build directory..."
  rm -rf "$BUILD_DIR"
fi

# Detect LibTorch path
CMAKE_PREFIX=""
if python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null; then
  CMAKE_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
  echo "LibTorch found via pip: $CMAKE_PREFIX"
elif [ -n "$LIBTORCH_DIR" ]; then
  CMAKE_PREFIX="$LIBTORCH_DIR"
  echo "LibTorch from LIBTORCH_DIR: $CMAKE_PREFIX"
elif [ -n "$CMAKE_PREFIX_PATH" ]; then
  echo "Using CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
else
  echo "Warning: LibTorch not found. Set CMAKE_PREFIX_PATH or install PyTorch."
  echo "  pip install torch  # CPU"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1"
  exit 1
fi

# Detect platform
echo ""
echo "=== llm-cpp Build ==="
echo "Platform: $(uname -s) $(uname -m)"

if [ -n "$FORCE_CPU" ]; then
  echo "Mode: CPU only (forced)"
elif [ -n "$FORCE_CUDA" ]; then
  echo "Mode: CUDA (forced)"
  if ! command -v nvcc &>/dev/null; then
    echo "Error: nvcc not found. Install CUDA Toolkit or load module."
    echo "  Jetstream: module load cuda/12.x"
    exit 1
  fi
  CUDA_VER=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+' 2>/dev/null || nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  echo "CUDA Version: $CUDA_VER"
  if [ -n "$CUDA_VER" ]; then
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$MAJOR" -lt 12 ]; then
      echo "Warning: CUDA $CUDA_VER detected. H100 requires CUDA 12.0+"
    fi
  fi

  # ── Resolve CUDA_HOME to a single path (avoids "conflicting CUDA installs" error) ──
  if [ -z "${CUDA_HOME:-}" ]; then
    # Prefer /usr/local/cuda (system CUDA), then try to find from nvcc
    if [ -d "/usr/local/cuda" ]; then
      export CUDA_HOME="/usr/local/cuda"
    else
      # Derive from nvcc location: /usr/bin/nvcc → /usr, /usr/local/cuda/bin/nvcc → /usr/local/cuda
      NVCC_PATH="$(which nvcc 2>/dev/null)"
      if [ -n "$NVCC_PATH" ]; then
        export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
      fi
    fi
  fi
  if [ -n "${CUDA_HOME:-}" ]; then
    echo "CUDA_HOME: $CUDA_HOME"
    # Force CMake to use only this CUDA — prevents "conflicting installs" when
    # conda and system both have CUDA headers
    export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"
    # If nvcc isn't in CUDA_HOME (e.g. system nvcc at /usr/bin), fall back
    if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
      unset CMAKE_CUDA_COMPILER CUDACXX
    fi
  fi

  # ── Auto-detect nvToolsExt (required by some PyTorch builds) ──
  if [ -z "${NVTOOLSEXT_PATH:-}" ]; then
    for candidate in "${CUDA_HOME:-/usr/local/cuda}" /usr/local/cuda /usr/lib/x86_64-linux-gnu /usr/local/cuda/targets/x86_64-linux; do
      if [ -f "$candidate/lib/libnvToolsExt.so" ] || [ -f "$candidate/lib64/libnvToolsExt.so" ]; then
        export NVTOOLSEXT_PATH="$candidate"
        echo "Found nvToolsExt: $candidate"
        break
      fi
    done
    # Check conda env
    if [ -z "${NVTOOLSEXT_PATH:-}" ] && [ -n "${CONDA_PREFIX:-}" ]; then
      if [ -f "$CONDA_PREFIX/lib/libnvToolsExt.so" ]; then
        export NVTOOLSEXT_PATH="$CONDA_PREFIX"
        echo "Found nvToolsExt in conda: $CONDA_PREFIX"
      fi
    fi
    if [ -z "${NVTOOLSEXT_PATH:-}" ]; then
      echo "Warning: nvToolsExt not found. If build fails, install with:"
      echo "  sudo apt-get install -y nvidia-cuda-toolkit"
      echo "  # or: conda install -c conda-forge cudatoolkit-dev"
    fi
  fi
else
  if command -v nvcc &>/dev/null; then
    echo "Mode: CUDA (auto-detected)"
    nvcc --version | grep "release"
  elif [ "$(uname -s)" = "Darwin" ]; then
    echo "Mode: Metal/MPS (Apple Silicon)"
  else
    echo "Mode: CPU"
  fi
fi

# Configure
echo ""
echo "Configuring..."
cmake -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  ${CMAKE_PREFIX:+-DCMAKE_PREFIX_PATH="$CMAKE_PREFIX"} \
  $EXTRA_CMAKE_ARGS \
  "$ROOT"

# Build
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo ""
echo "Building with $NPROC threads..."
cmake --build "$BUILD_DIR" -j "$NPROC"

echo ""
echo "=== Build Complete ==="
echo ""
echo "Binaries in $BUILD_DIR/:"
for bin in olmo_train chat prepare_data convert_checkpoint convert_hf inspect_tokens mine_patterns benchmark_tokenizer dump_params \
           zwt_pretrain zwt_train zwt_export_hf zwt_numerical_audit \
           zwt_kernel_tests zwt_ddp_bucket_tests zwt_ddp_loopback_tests \
           zwt_tp_tests zwt_flash_attn_tests zwt_graph_bench \
           zwt_wgmma_bench zwt_wgmma_tests; do
  if [ -f "$BUILD_DIR/$bin" ]; then
    SIZE=$(du -h "$BUILD_DIR/$bin" | cut -f1)
    echo "  $bin  ($SIZE)"
  fi
done

echo ""
if [ -n "$FORCE_CUDA" ] || command -v nvcc &>/dev/null; then
echo "CUDA kernels built:"
echo "  - rms_norm.cu     (vectorized RMSNorm, warp reductions)"
echo "  - silu_mul.cu     (fused SiLU(gate) × up, zero intermediate alloc)"
echo "  - rope.cu         (fused RoPE + fused Q+K RoPE)"
echo "  - residual+norm   (fused residual add + RMSNorm, single pass)"
echo ""
fi

echo "Quick start:"
echo "  # Download tokenizer (one-time)"
echo "  mkdir -p data/gpt2"
echo "  curl -sL https://huggingface.co/gpt2/resolve/main/vocab.json -o data/gpt2/vocab.json"
echo "  curl -sL https://huggingface.co/gpt2/resolve/main/merges.txt -o data/gpt2/merges.txt"
echo ""
echo "  # Prepare data"
echo "  $BUILD_DIR/prepare_data --download-hf roneneldan/TinyStories --output data/tokens.npy \\"
echo "    --vocab-file data/gpt2/vocab.json --merges-file data/gpt2/merges.txt"
echo ""
echo "  # Train"
echo "  $BUILD_DIR/olmo_train --train --config configs/olmo2_125M.json \\"
echo "    --data-path data/tokens.npy --device auto --steps 5000 --lr 3e-4"
