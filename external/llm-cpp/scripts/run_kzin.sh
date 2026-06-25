#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# run_kzin.sh — Full benchmark suite for RTX 3060 (12GB VRAM)
#
# Adapted from jetstream_run.sh for the kzin Linux system.
# Runs the full 3-way comparison: Python OLMo-core vs cpp-llm vs llm-cpp
#
# Usage:
#   export CPP_LLM_DIR=/path/to/cpp-llm   # REQUIRED: already-cloned cpp-llm
#   ./scripts/run_kzin.sh
#
# What it does:
#   1. Verifies prerequisites (CUDA, cmake, python3, git)
#   2. Builds both C++ projects with CUDA for RTX 3060 (sm_86)
#   3. Sets up Python venv with PyTorch+CUDA and OLMo-core
#   4. Downloads GPT-2 tokenizer + TinyStories data
#   5. Runs 3-way training benchmark (30M, 125M, 350M)
#   6. Trains models for generation quality evaluation
#   7. Runs 10-prompt chat evaluation with automated grading
#   8. Produces final report in results/
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="$REPO_ROOT/results"
CHECKPOINT_DIR="$REPO_ROOT/checkpoints"
DEVICE="cuda"
SEED=42

# RTX 3060 = Ampere sm_86, 12GB VRAM
CUDA_ARCH="86"

# Benchmark settings (tuned for 12GB VRAM)
BENCH_STEPS=100
BENCH_WARMUP=5
BENCH_BATCH=4
BENCH_SEQ=256

# 350M needs smaller batch on 12GB
BENCH_350_BATCH=2
BENCH_350_SEQ=256

# Training for chat quality (longer run)
CHAT_TRAIN_STEPS=2000
CHAT_BATCH=8
CHAT_SEQ=256

# ── Parse arguments ──
CLEAN=""
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=1 ;;
        --help|-h)
            echo "Usage: CPP_LLM_DIR=/path/to/cpp-llm $0 [--clean]"
            echo "  --clean   Force rebuild of all binaries"
            exit 0
            ;;
    esac
done

# Clean builds if requested
if [ -n "$CLEAN" ]; then
    echo "Cleaning builds..."
    rm -rf "$REPO_ROOT/build" "$REPO_ROOT/.venv" "$REPO_ROOT/results"
fi

# ── cpp-llm location (must be set) ──
if [ -z "${CPP_LLM_DIR:-}" ]; then
    echo "ERROR: CPP_LLM_DIR is not set."
    echo ""
    echo "Set it to the path of your already-cloned cpp-llm repository:"
    echo "  export CPP_LLM_DIR=/path/to/cpp-llm"
    echo "  ./scripts/run_kzin.sh"
    exit 1
fi

if [ ! -d "$CPP_LLM_DIR" ]; then
    echo "ERROR: CPP_LLM_DIR=$CPP_LLM_DIR does not exist."
    exit 1
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
section() { echo -e "\n${BOLD}${BLUE}═══ $* ═══${NC}\n"; }

# ── 1. Check Prerequisites ──
check_prerequisites() {
    section "Checking prerequisites"

    command -v git >/dev/null || err "git not found"
    log "git: $(git --version | head -1)"

    command -v cmake >/dev/null || err "cmake not found"
    log "cmake: $(cmake --version | head -1)"

    command -v python3 >/dev/null || err "python3 not found"
    log "python3: $(python3 --version)"

    command -v nvcc >/dev/null || err "nvcc not found — CUDA toolkit required"
    log "nvcc: $(nvcc --version | grep release)"

    # Check CUDA device
    if command -v nvidia-smi >/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "GPU: $GPU_NAME (${GPU_MEM}MB)"

        if [ "${GPU_MEM:-0}" -lt 10000 ]; then
            warn "GPU has less than 10GB VRAM — 350M model may OOM"
        fi
    fi

    log "cpp-llm: $CPP_LLM_DIR"
}

# ── 2. Build llm-cpp (this repo) ──
build_llm_cpp() {
    section "Building llm-cpp (Experimental)"

    # Skip if all binaries already exist
    if [ -f "$REPO_ROOT/build/olmo_train" ] && \
       [ -f "$REPO_ROOT/build/chat" ] && \
       [ -f "$REPO_ROOT/build/prepare_data" ]; then
        log "llm-cpp already built — skipping (use --clean to force rebuild)"
        log "  olmo_train:   $(du -h "$REPO_ROOT/build/olmo_train" | cut -f1)"
        log "  chat:         $(du -h "$REPO_ROOT/build/chat" | cut -f1)"
        log "  prepare_data: $(du -h "$REPO_ROOT/build/prepare_data" | cut -f1)"
        return
    fi

    cd "$REPO_ROOT"

    # Auto-detect LibTorch from pip
    CMAKE_PREFIX=""
    if python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null; then
        CMAKE_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
        log "LibTorch found: $CMAKE_PREFIX"
    else
        err "PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    fi

    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        2>&1 | tail -10

    make -j"$(nproc)" 2>&1 | tail -10

    [ -f "$REPO_ROOT/build/olmo_train" ] || err "olmo_train build failed"
    [ -f "$REPO_ROOT/build/chat" ] || err "chat build failed"
    [ -f "$REPO_ROOT/build/prepare_data" ] || err "prepare_data build failed"

    log "Built: olmo_train, chat, prepare_data"
    cd "$REPO_ROOT"
}

# ── 3. Build cpp-llm ──
build_cpp_llm() {
    section "Building cpp-llm"

    # Check if cpp-llm is already built
    CPP_LLM_TRAIN=""
    for name in olmo_train train main; do
        if [ -f "$CPP_LLM_DIR/build/$name" ]; then
            CPP_LLM_TRAIN="$CPP_LLM_DIR/build/$name"
            break
        fi
    done

    CPP_LLM_CHAT=""
    if [ -f "$CPP_LLM_DIR/build/chat" ]; then
        CPP_LLM_CHAT="$CPP_LLM_DIR/build/chat"
    fi

    # Skip if training binary already exists
    if [ -n "$CPP_LLM_TRAIN" ]; then
        log "cpp-llm already built — skipping (use --clean to force rebuild)"
        log "  train: $CPP_LLM_TRAIN ($(du -h "$CPP_LLM_TRAIN" | cut -f1))"
        [ -n "$CPP_LLM_CHAT" ] && log "  chat:  $CPP_LLM_CHAT ($(du -h "$CPP_LLM_CHAT" | cut -f1))"
        return
    fi

    # Auto-detect LibTorch from pip
    CMAKE_PREFIX=""
    if python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null; then
        CMAKE_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
    fi

    cd "$CPP_LLM_DIR"
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        ${CMAKE_PREFIX:+-DCMAKE_PREFIX_PATH="$CMAKE_PREFIX"} \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        2>&1 | tail -10

    make -j"$(nproc)" 2>&1 | tail -10

    # Find the training binary
    CPP_LLM_TRAIN=""
    for name in olmo_train train main; do
        if [ -f "$CPP_LLM_DIR/build/$name" ]; then
            CPP_LLM_TRAIN="$CPP_LLM_DIR/build/$name"
            break
        fi
    done

    if [ -n "$CPP_LLM_TRAIN" ]; then
        log "Built cpp-llm: $CPP_LLM_TRAIN"
    else
        warn "cpp-llm training binary not found"
    fi

    # Find chat binary
    CPP_LLM_CHAT=""
    if [ -f "$CPP_LLM_DIR/build/chat" ]; then
        CPP_LLM_CHAT="$CPP_LLM_DIR/build/chat"
        log "Built cpp-llm chat: $CPP_LLM_CHAT"
    fi

    cd "$REPO_ROOT"
}

# ── 4. Setup Python Environment ──
setup_python_env() {
    section "Setting up Python environment"

    if [ -d "$VENV_DIR" ] && "$VENV_DIR/bin/python" -c "import torch" 2>/dev/null; then
        log "Existing venv found with PyTorch"
    else
        log "Creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"

        log "Installing PyTorch with CUDA..."
        "$VENV_DIR/bin/pip" install --upgrade pip -q

        # Try CUDA 12.4, then 12.1, then CPU fallback
        "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124 -q \
            || "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121 -q \
            || "$VENV_DIR/bin/pip" install torch -q

        log "Installing OLMo-core..."
        "$VENV_DIR/bin/pip" install ai2-olmo-core -q \
            || warn "OLMo-core install failed (will use fallback model)"

        "$VENV_DIR/bin/pip" install numpy -q
    fi

    if "$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log "PyTorch CUDA verified in venv"
    else
        warn "PyTorch CUDA not available in venv — Python benchmarks will be slow"
    fi
}

# ── 5. Prepare Data ──
prepare_data() {
    section "Preparing data"

    mkdir -p "$DATA_DIR" "$DATA_DIR/gpt2"

    # Download GPT-2 tokenizer files
    if [ ! -f "$DATA_DIR/gpt2/vocab.json" ]; then
        log "Downloading GPT-2 vocab.json..."
        curl -sL "https://huggingface.co/gpt2/resolve/main/vocab.json" \
            -o "$DATA_DIR/gpt2/vocab.json"
    fi
    if [ ! -f "$DATA_DIR/gpt2/merges.txt" ]; then
        log "Downloading GPT-2 merges.txt..."
        curl -sL "https://huggingface.co/gpt2/resolve/main/merges.txt" \
            -o "$DATA_DIR/gpt2/merges.txt"
    fi
    log "GPT-2 tokenizer: $DATA_DIR/gpt2/"

    # Prepare tokenized data
    if [ -f "$DATA_DIR/tinystories_gpt2.npy" ]; then
        log "Data already exists: $DATA_DIR/tinystories_gpt2.npy"
    else
        log "Downloading and tokenizing TinyStories..."

        if [ -f "$REPO_ROOT/build/prepare_data" ]; then
            "$REPO_ROOT/build/prepare_data" \
                --download-hf roneneldan/TinyStories \
                --output "$DATA_DIR/tinystories_gpt2.npy" \
                --vocab-file "$DATA_DIR/gpt2/vocab.json" \
                --merges-file "$DATA_DIR/gpt2/merges.txt" \
                --threads "$(nproc)" \
                2>&1 | tail -5
        else
            warn "prepare_data not found, generating random tokens"
            "$REPO_ROOT/build/prepare_data" \
                --random 100000000 \
                --output "$DATA_DIR/tinystories_gpt2.npy" \
                2>&1 | tail -3
        fi
        log "Data prepared: $DATA_DIR/tinystories_gpt2.npy"
    fi
}

# ── 6. Generate JSON configs for chat binary ──
generate_json_configs() {
    section "Generating JSON configs for chat"

    mkdir -p "$REPO_ROOT/configs"

    for conf_file in "$REPO_ROOT"/conf/olmo*.conf; do
        base=$(basename "$conf_file" .conf)
        json_out="$REPO_ROOT/configs/${base}.json"
        python3 "$SCRIPT_DIR/conf_to_json.py" "$conf_file" "$json_out"
        log "Generated $json_out"
    done
}

# ── 7. Run Training Benchmark ──
run_training_benchmark() {
    section "Running 3-way training benchmark"

    mkdir -p "$RESULTS_DIR"

    # ── 30M benchmark ──
    log "Benchmarking 30M model..."
    "$VENV_DIR/bin/python" "$SCRIPT_DIR/benchmark.py" \
        --model 30M \
        --device "$DEVICE" \
        --seed "$SEED" \
        --steps "$BENCH_STEPS" \
        --all-variants \
        --bpe-vocab "$DATA_DIR/gpt2/vocab.json" \
        --cpp-llm-dir "$CPP_LLM_DIR" \
        2>&1 | tee "$RESULTS_DIR/bench_30M.log"

    cp -f benchmark_results.json "$RESULTS_DIR/bench_30M.json" 2>/dev/null || true

    # ── 125M benchmark ──
    log "Benchmarking 125M model..."
    "$VENV_DIR/bin/python" "$SCRIPT_DIR/benchmark.py" \
        --model 100M \
        --device "$DEVICE" \
        --seed "$SEED" \
        --steps "$BENCH_STEPS" \
        --all-variants \
        --bpe-vocab "$DATA_DIR/gpt2/vocab.json" \
        --cpp-llm-dir "$CPP_LLM_DIR" \
        2>&1 | tee "$RESULTS_DIR/bench_125M.log"

    cp -f benchmark_results.json "$RESULTS_DIR/bench_125M.json" 2>/dev/null || true

    # ── 350M benchmark (llm-cpp — reduced batch for 12GB VRAM) ──
    log "Benchmarking 350M model (llm-cpp, batch=${BENCH_350_BATCH} for 12GB VRAM)..."

    BENCH_350_CONF=$(mktemp /tmp/olmo_350M_bench.XXXXXX.conf)
    cat > "$BENCH_350_CONF" << CONFEOF
[model]
d_model	1024
vocab_size	50257
n_layers	24
n_heads	16
n_kv_heads	-1
head_dim	-1
rope_theta	500000
layer_norm_eps	1e-6
init_std	0.02
use_qk_norm	1
num_mtp_heads	2
mtp_loss_weight	0.1

[training]
steps	${BENCH_STEPS}
batch_size	${BENCH_350_BATCH}
seq_len	${BENCH_350_SEQ}
lr	3e-4
warmup_steps	${BENCH_WARMUP}
grad_accum	1
optimizer	muon
amp	1
seed	${SEED}
profile	1
save

[data]
data_path	${DATA_DIR}/tinystories_gpt2.npy
bpe_vocab	${DATA_DIR}/gpt2/vocab.json

[optimization]
fused	1
mup	1
multi_res	1

[device]
device	cuda
CONFEOF

    BENCH_START=$(date +%s%N)
    "$REPO_ROOT/build/olmo_train" "$BENCH_350_CONF" \
        2>&1 | tee "$RESULTS_DIR/bench_350M_experimental.log"
    BENCH_END=$(date +%s%N)
    BENCH_MS=$(( (BENCH_END - BENCH_START) / 1000000 ))
    log "350M benchmark (Experimental): ${BENCH_MS}ms for ${BENCH_STEPS} steps"
    rm -f "$BENCH_350_CONF"

    # 350M with cpp-llm
    if [ -n "${CPP_LLM_TRAIN:-}" ] && [ -f "${CPP_LLM_TRAIN:-}" ]; then
        log "Benchmarking 350M model (cpp-llm)..."
        BENCH_START=$(date +%s%N)
        "$CPP_LLM_TRAIN" --train \
            --data-path "$DATA_DIR/tinystories_gpt2.npy" \
            --device cuda \
            --batch-size "$BENCH_350_BATCH" \
            --seq-len "$BENCH_350_SEQ" \
            --steps "$BENCH_STEPS" \
            --lr 3e-4 \
            --optimizer muon \
            2>&1 | tee "$RESULTS_DIR/bench_350M_cpplm.log"
        BENCH_END=$(date +%s%N)
        BENCH_MS=$(( (BENCH_END - BENCH_START) / 1000000 ))
        log "350M benchmark (cpp-llm): ${BENCH_MS}ms for ${BENCH_STEPS} steps"
    fi
}

# ── 8. Train Models for Chat Evaluation ──
train_for_chat() {
    section "Training models for chat evaluation"

    mkdir -p "$CHECKPOINT_DIR"

    # Train 30M llm-cpp
    CHAT_CONF=$(mktemp /tmp/olmo_30M_chat.XXXXXX.conf)
    cat > "$CHAT_CONF" << CONFEOF
[model]
d_model	256
vocab_size	50257
n_layers	4
n_heads	8
n_kv_heads	-1
head_dim	-1
rope_theta	500000
layer_norm_eps	1e-6
init_std	0.02
use_qk_norm	1
num_mtp_heads	0
mtp_loss_weight	0.0

[training]
steps	${CHAT_TRAIN_STEPS}
batch_size	${CHAT_BATCH}
seq_len	${CHAT_SEQ}
lr	3e-4
warmup_steps	100
grad_accum	1
optimizer	muon
amp	0
seed	${SEED}
profile	0
save	${CHECKPOINT_DIR}/30M_experimental.pt

[data]
data_path	${DATA_DIR}/tinystories_gpt2.npy
bpe_vocab	${DATA_DIR}/gpt2/vocab.json

[optimization]
fused	0
mup	1
multi_res	0

[device]
device	cuda
CONFEOF

    log "Training 30M for chat (llm-cpp Experimental, ${CHAT_TRAIN_STEPS} steps)..."
    "$REPO_ROOT/build/olmo_train" "$CHAT_CONF" \
        2>&1 | tee "$RESULTS_DIR/train_30M_experimental.log"
    rm -f "$CHAT_CONF"

    # Train 30M with cpp-llm
    if [ -n "${CPP_LLM_TRAIN:-}" ] && [ -f "${CPP_LLM_TRAIN:-}" ]; then
        log "Training 30M for chat (cpp-llm, ${CHAT_TRAIN_STEPS} steps)..."
        "$CPP_LLM_TRAIN" --train \
            --data-path "$DATA_DIR/tinystories_gpt2.npy" \
            --device cuda \
            --batch-size "$CHAT_BATCH" \
            --seq-len "$CHAT_SEQ" \
            --steps "$CHAT_TRAIN_STEPS" \
            --lr 3e-4 \
            --warmup-steps 100 \
            --optimizer muon \
            --save "$CHECKPOINT_DIR/30M_cpplm.pt" \
            2>&1 | tee "$RESULTS_DIR/train_30M_cpplm.log"
    fi

    # Train 30M Python model
    log "Training 30M for chat (Python OLMo-core, ${CHAT_TRAIN_STEPS} steps)..."
    CHAT_TRAIN_STEPS=$CHAT_TRAIN_STEPS \
    CHAT_BATCH=$CHAT_BATCH \
    CHAT_SEQ=$CHAT_SEQ \
    DATA_PATH="$DATA_DIR/tinystories_gpt2.npy" \
    SAVE_PATH="$CHECKPOINT_DIR/30M_python.pt" \
    "$VENV_DIR/bin/python" - << 'PYEOF'
import json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
STEPS = int(os.environ.get("CHAT_TRAIN_STEPS", "2000"))
BATCH = int(os.environ.get("CHAT_BATCH", "8"))
SEQ = int(os.environ.get("CHAT_SEQ", "256"))
LR = 3e-4
DATA_PATH = os.environ.get("DATA_PATH", "data/tinystories_gpt2.npy")
SAVE_PATH = os.environ.get("SAVE_PATH", "checkpoints/30M_python.pt")

torch.manual_seed(SEED)

# Try OLMo-core first, fallback to vanilla PyTorch
try:
    from olmo_core.nn.transformer import TransformerConfig, Transformer
    config = TransformerConfig(d_model=256, n_layers=4, n_heads=8, vocab_size=50257)
    model = config.build(device=DEVICE)
    print("Using OLMo-core Transformer")
except Exception:
    class MiniTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50257, 256)
            layer = nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True,
                                                norm_first=True, dropout=0.0)
            self.encoder = nn.TransformerEncoder(layer, 4)
            self.ln = nn.LayerNorm(256)
            self.lm_head = nn.Linear(256, 50257, bias=False)
        def forward(self, x, labels=None):
            h = self.embed(x)
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            h = self.encoder(h, mask=mask, is_causal=True)
            h = self.ln(h)
            logits = self.lm_head(h)
            if labels is not None:
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50257),
                                       labels[:, 1:].reshape(-1))
                return logits, loss
            return logits
    model = MiniTransformer().to(DEVICE)
    print("Using fallback MiniTransformer")

# Load data
if os.path.exists(DATA_PATH):
    tokens = np.load(DATA_PATH).astype(np.int64)
    print(f"Loaded {len(tokens):,} tokens from {DATA_PATH}")
else:
    tokens = np.random.randint(0, 50257, size=10_000_000, dtype=np.int64)
    print("Using random tokens")

data = torch.from_numpy(tokens)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Training loop
model.train()
start = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    batch = torch.stack([data[i:i+SEQ+1] for i in idx]).to(DEVICE)
    x, y = batch[:, :-1], batch[:, 1:]

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    loss = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0 or step == 1:
        elapsed = time.time() - start
        tok_s = step * BATCH * SEQ / elapsed
        print(f"step {step}/{STEPS} | loss {loss.item():.4f} | {tok_s:.0f} tok/s")

elapsed = time.time() - start
print(f"Training done: {STEPS} steps in {elapsed:.1f}s")

# Save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved: {SAVE_PATH}")
PYEOF
}

# ── 9. Run Chat Evaluation ──
run_chat_evaluation() {
    section "Running chat evaluation (10 prompts)"

    generate_json_configs

    EVAL_ARGS=(
        --vocab-file "$DATA_DIR/gpt2/vocab.json"
        --merges-file "$DATA_DIR/gpt2/merges.txt"
        --device "$DEVICE"
        --output "$RESULTS_DIR/generation_eval.json"
    )

    # llm-cpp (Experimental)
    if [ -f "$CHECKPOINT_DIR/30M_experimental.pt" ] && [ -f "$REPO_ROOT/build/chat" ]; then
        EVAL_ARGS+=(
            --llm-cpp-chat "$REPO_ROOT/build/chat"
            --llm-cpp-checkpoint "$CHECKPOINT_DIR/30M_experimental.pt"
            --llm-cpp-config "$REPO_ROOT/configs/olmo.json"
        )
    fi

    # cpp-llm
    if [ -f "$CHECKPOINT_DIR/30M_cpplm.pt" ] && [ -n "${CPP_LLM_CHAT:-}" ] && [ -f "${CPP_LLM_CHAT:-}" ]; then
        CPP_LLM_CFG=""
        for cfg_path in "$CPP_LLM_DIR/configs/olmo2_30M.json" \
                        "$CPP_LLM_DIR/configs/30M.json" \
                        "$REPO_ROOT/configs/olmo.json"; do
            if [ -f "$cfg_path" ]; then
                CPP_LLM_CFG="$cfg_path"
                break
            fi
        done
        if [ -n "$CPP_LLM_CFG" ]; then
            EVAL_ARGS+=(
                --cpp-llm-chat "$CPP_LLM_CHAT"
                --cpp-llm-checkpoint "$CHECKPOINT_DIR/30M_cpplm.pt"
                --cpp-llm-config "$CPP_LLM_CFG"
            )
        fi
    fi

    # Python
    PYTHON_CONFIG='{"d_model":256,"vocab_size":50257,"n_layers":4,"n_heads":8}'
    EVAL_ARGS+=(
        --python-config "$PYTHON_CONFIG"
    )
    if [ -f "$CHECKPOINT_DIR/30M_python.pt" ]; then
        EVAL_ARGS+=(--python-checkpoint "$CHECKPOINT_DIR/30M_python.pt")
    fi

    "$VENV_DIR/bin/python" "$SCRIPT_DIR/eval_generation.py" "${EVAL_ARGS[@]}" \
        2>&1 | tee "$RESULTS_DIR/generation_eval.log"
}

# ── 10. Generate Final Report ──
generate_report() {
    section "Generating final report"

    REPORT="$RESULTS_DIR/REPORT.txt"
    cat > "$REPORT" << 'HEADER'
═══════════════════════════════════════════════════════════════
  llm-cpp Benchmark Report
  3-Way Comparison: Python OLMo-core vs cpp-llm vs llm-cpp
  Target: NVIDIA RTX 3060 (12GB VRAM)
═══════════════════════════════════════════════════════════════
HEADER

    echo "" >> "$REPORT"
    echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$REPORT"
    echo "Host: $(hostname)" >> "$REPORT"

    if command -v nvidia-smi >/dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" >> "$REPORT"
        echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)" >> "$REPORT"
        echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}')" >> "$REPORT"
        echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" >> "$REPORT"
    fi

    echo "" >> "$REPORT"

    # Append benchmark logs
    echo "── Training Benchmark (30M) ──" >> "$REPORT"
    if [ -f "$RESULTS_DIR/bench_30M.log" ]; then
        grep -E '(tok/s|speedup|slower|faster|Framework)' "$RESULTS_DIR/bench_30M.log" >> "$REPORT" 2>/dev/null || true
    fi

    echo "" >> "$REPORT"
    echo "── Training Benchmark (125M) ──" >> "$REPORT"
    if [ -f "$RESULTS_DIR/bench_125M.log" ]; then
        grep -E '(tok/s|speedup|slower|faster|Framework)' "$RESULTS_DIR/bench_125M.log" >> "$REPORT" 2>/dev/null || true
    fi

    echo "" >> "$REPORT"
    echo "── Training Benchmark (350M) ──" >> "$REPORT"
    if [ -f "$RESULTS_DIR/bench_350M_experimental.log" ]; then
        grep -E '(tok/s|params|Backend|Model|Throughput)' "$RESULTS_DIR/bench_350M_experimental.log" >> "$REPORT" 2>/dev/null || true
    fi

    echo "" >> "$REPORT"
    echo "── Generation Quality ──" >> "$REPORT"
    if [ -f "$RESULTS_DIR/generation_eval.log" ]; then
        sed -n '/^SUMMARY$/,/^$/p' "$RESULTS_DIR/generation_eval.log" >> "$REPORT" 2>/dev/null || true
    fi

    echo "" >> "$REPORT"
    echo "── Files ──" >> "$REPORT"
    echo "Benchmark JSONs: $RESULTS_DIR/bench_*.json" >> "$REPORT"
    echo "Generation eval: $RESULTS_DIR/generation_eval.json" >> "$REPORT"
    echo "Training logs:   $RESULTS_DIR/train_*.log" >> "$REPORT"
    echo "" >> "$REPORT"

    log "Report: $REPORT"
    echo ""
    cat "$REPORT"
}

# ── Main ──
main() {
    echo -e "${BOLD}"
    echo "  ╔═══════════════════════════════════════════════════╗"
    echo "  ║  llm-cpp: Full Benchmark Suite (RTX 3060)        ║"
    echo "  ║  Python OLMo-core × cpp-llm × llm-cpp            ║"
    echo "  ╚═══════════════════════════════════════════════════╝"
    echo -e "${NC}"

    TOTAL_START=$(date +%s)

    check_prerequisites
    build_llm_cpp
    build_cpp_llm
    setup_python_env
    prepare_data
    run_training_benchmark
    train_for_chat
    run_chat_evaluation
    generate_report

    TOTAL_END=$(date +%s)
    TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))

    echo ""
    log "All done in ${TOTAL_MIN} minutes."
    log "Results: $RESULTS_DIR/"
}

main "$@"
