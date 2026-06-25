#!/bin/bash
# =============================================================================
# Jetstream H100 Benchmark: C++ vs Python, 125M transformer
#
# This script:
#   1. Rebuilds C++ binaries with CUDA
#   2. Prepares tokenized data on the media volume
#   3. Runs C++ training (ForeachAdamW + GPU-resident data + fused model)
#   4. Runs Python PyTorch training (same architecture, same data)
#   5. Profiles both with nsys (NVIDIA Nsight Systems)
#   6. Copies profiles to a pickup directory for download
#
# Usage:
#   chmod +x scripts/jetstream_benchmark.sh
#   ./scripts/jetstream_benchmark.sh
#
# Prerequisites:
#   - CUDA 12.x + nvcc on PATH
#   - pip install torch (CUDA build)
#   - nsys (NVIDIA Nsight Systems) on PATH
# =============================================================================

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME="/media/volume/Prep_and_Voice_Training"
DATA_DIR="${VOLUME}/llm_benchmark_data"
RESULTS_DIR="${VOLUME}/llm_benchmark_results"
STEPS=100
BATCH_SIZE=8
SEQ_LEN=512

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "  OLMo C++ vs Python Benchmark — 125M Transformer"
echo "  Jetstream H100"
echo "============================================================"
echo ""

# ── 0. Verify environment ──────────────────────────────────────────────────
info "Checking environment..."

command -v nvcc    >/dev/null 2>&1 || fail "nvcc not found. Load CUDA module: module load cuda/12.x"
command -v cmake   >/dev/null 2>&1 || fail "cmake not found"
command -v python3 >/dev/null 2>&1 || fail "python3 not found"
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || fail "PyTorch CUDA not available"

CUDA_VER=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+' || echo "unknown")
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")

ok "CUDA $CUDA_VER | GPU: $GPU_NAME | PyTorch $TORCH_VER"

# Ensure numpy is installed (needed by Python benchmark)
# On externally-managed environments (PEP 668), use a venv
if ! python3 -c "import numpy" 2>/dev/null; then
    info "Installing numpy..."
    if pip install numpy --quiet 2>/dev/null; then
        ok "numpy installed"
    else
        info "System pip blocked (PEP 668), creating venv..."
        VENV_DIR="${REPO_DIR}/.venv"
        if [ ! -d "$VENV_DIR" ]; then
            python3 -m venv "$VENV_DIR" --system-site-packages
        fi
        source "$VENV_DIR/bin/activate"
        pip install numpy --quiet
        ok "numpy installed in venv"
    fi
elif [ -d "${REPO_DIR}/.venv" ]; then
    # Reuse existing venv if it exists (keeps numpy + torch visible)
    source "${REPO_DIR}/.venv/bin/activate"
    ok "Using existing venv"
fi

# Check nsys
if command -v nsys >/dev/null 2>&1; then
    HAVE_NSYS=1
    NSYS_VER=$(nsys --version 2>&1 | head -1 || echo "unknown")
    ok "nsys found: $NSYS_VER"
    # Older nsys (< 2023) doesn't support --cuda-memory-usage or --stats
    NSYS_EXTRA=""
    if nsys profile --help 2>&1 | grep -q "cuda-memory-usage"; then
        NSYS_EXTRA="--cuda-memory-usage=true --stats=true"
    fi
else
    HAVE_NSYS=0
    warn "nsys not found — skipping profiling. Install NVIDIA Nsight Systems for GPU profiles."
fi

# Check volume mount
if [ ! -d "$VOLUME" ]; then
    warn "Volume $VOLUME not found, using /tmp instead"
    VOLUME="/tmp"
    DATA_DIR="${VOLUME}/llm_benchmark_data"
    RESULTS_DIR="${VOLUME}/llm_benchmark_results"
fi

mkdir -p "$DATA_DIR" "$RESULTS_DIR"
ok "Data dir: $DATA_DIR"
ok "Results dir: $RESULTS_DIR"

# ── 1. Build C++ ───────────────────────────────────────────────────────────
echo ""
info "=== Phase 1: Building C++ binaries ==="
cd "$REPO_DIR"
./scripts/build.sh --cuda 2>&1 | tail -20
ok "C++ build complete"

# ── 2. Prepare data ───────────────────────────────────────────────────────
echo ""
info "=== Phase 2: Preparing tokenized data ==="

TOKEN_FILE="${DATA_DIR}/tinystories_gpt2.npy"

if [ -f "$TOKEN_FILE" ]; then
    ok "Token file already exists: $TOKEN_FILE"
else
    info "Downloading tokenizer files..."
    mkdir -p "${DATA_DIR}/gpt2"
    curl -sL "https://huggingface.co/gpt2/resolve/main/vocab.json" -o "${DATA_DIR}/gpt2/vocab.json"
    curl -sL "https://huggingface.co/gpt2/resolve/main/merges.txt"  -o "${DATA_DIR}/gpt2/merges.txt"
    ok "Tokenizer downloaded"

    info "Tokenizing TinyStories dataset (this may take a few minutes)..."
    "${REPO_DIR}/build/prepare_data" \
        --download-hf roneneldan/TinyStories \
        --output "$TOKEN_FILE" \
        --vocab-file "${DATA_DIR}/gpt2/vocab.json" \
        --merges-file "${DATA_DIR}/gpt2/merges.txt" \
        2>&1 | tail -5
    ok "Data prepared: $TOKEN_FILE"
fi

# Create the benchmark conf with actual data path
BENCH_CONF="${RESULTS_DIR}/benchmark_125M.conf"
sed "s|DATA_PATH_PLACEHOLDER|${TOKEN_FILE}|" "${REPO_DIR}/conf/benchmark_125M.conf" > "$BENCH_CONF"
ok "Config written: $BENCH_CONF"

# ── 3. C++ benchmark ─────────────────────────────────────────────────────
echo ""
info "=== Phase 3: C++ Training Benchmark (${STEPS} steps) ==="
CPP_LOG="${RESULTS_DIR}/cpp_benchmark.log"

if [ "$HAVE_NSYS" -eq 1 ]; then
    CPP_NSYS="${RESULTS_DIR}/cpp_profile"
    info "Running with nsys profiling..."
    nsys profile \
        --output "$CPP_NSYS" \
        --force-overwrite true \
        --trace cuda,nvtx,osrt \
        $NSYS_EXTRA \
        "${REPO_DIR}/build/olmo_train" "$BENCH_CONF" \
        2>&1 | tee "$CPP_LOG"
    ok "C++ nsys profile saved to ${RESULTS_DIR}/"
else
    info "Running without profiling..."
    "${REPO_DIR}/build/olmo_train" "$BENCH_CONF" 2>&1 | tee "$CPP_LOG"
fi
ok "C++ benchmark complete"

# ── 4. Python benchmark ──────────────────────────────────────────────────
echo ""
info "=== Phase 4: Python PyTorch Benchmark (${STEPS} steps) ==="
PY_LOG="${RESULTS_DIR}/python_benchmark.log"

if [ "$HAVE_NSYS" -eq 1 ]; then
    PY_NSYS="${RESULTS_DIR}/python_profile"
    info "Running with nsys profiling..."
    nsys profile \
        --output "$PY_NSYS" \
        --force-overwrite true \
        --trace cuda,nvtx,osrt \
        $NSYS_EXTRA \
        python3 "${REPO_DIR}/scripts/py_benchmark_125M.py" \
            --data-path "$TOKEN_FILE" \
            --steps "$STEPS" \
            --batch-size "$BATCH_SIZE" \
            --seq-len "$SEQ_LEN" \
        2>&1 | tee "$PY_LOG"
    ok "Python nsys profile saved to ${RESULTS_DIR}/"
else
    info "Running without profiling..."
    python3 "${REPO_DIR}/scripts/py_benchmark_125M.py" \
        --data-path "$TOKEN_FILE" \
        --steps "$STEPS" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        2>&1 | tee "$PY_LOG"
fi
ok "Python benchmark complete"

# ── 5. Summary ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  BENCHMARK RESULTS"
echo "============================================================"
echo ""
echo "--- C++ (ForeachAdamW + GPU data + fused model) ---"
grep -E "(Wall time|Throughput|step_ms|tok/s|Training Summary)" "$CPP_LOG" | tail -10
echo ""
echo "--- Python (PyTorch AdamW + GPU data + fused QKV) ---"
grep -E "(Wall time|Throughput|Avg step|tok/s|Training Summary)" "$PY_LOG" | tail -10
echo ""
echo "============================================================"
echo ""

# ── 6. Copy results for download ─────────────────────────────────────────
PICKUP_DIR="${RESULTS_DIR}/pickup"
mkdir -p "$PICKUP_DIR"

cp "$CPP_LOG" "$PICKUP_DIR/"
cp "$PY_LOG" "$PICKUP_DIR/"
if [ "$HAVE_NSYS" -eq 1 ]; then
    # Copy all profile formats (nsys-rep, qdstrm, sqlite) — older nsys versions
    # produce .qdstrm instead of .nsys-rep
    for ext in nsys-rep qdstrm sqlite; do
        cp "${CPP_NSYS}.${ext}" "$PICKUP_DIR/" 2>/dev/null || true
        cp "${PY_NSYS}.${ext}"  "$PICKUP_DIR/" 2>/dev/null || true
    done
fi

echo "All results in: ${PICKUP_DIR}/"
ls -lh "$PICKUP_DIR/"
echo ""
echo "To download to your local machine:"
echo "  scp -r <jetstream-host>:${PICKUP_DIR}/ ./benchmark_results/"
echo ""
ok "Benchmark complete!"
