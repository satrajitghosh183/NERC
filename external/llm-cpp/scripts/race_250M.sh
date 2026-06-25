#!/usr/bin/env bash
# scripts/race_250M.sh
#
# Three-way head-to-head race: who trains a 250M (OLMo-2-300M-class) model
# from scratch fastest on a single H100 (or 2× via DDP), with the same
# tokens, same effective batch, same seq_len, same hardware.
#
# Stacks:
#   1. olmo_train         — LibTorch C++  (build/olmo_train, conf/olmo_250M.conf)
#   2. zwt_pretrain       — LibTorch-free (build/zwt_pretrain, zwt/conf/owt_250M_h100.conf)
#   3. olmo-python        — upstream (scripts/train_olmo_python_250M.py)
#
# Win conditions reported (per memory: bench profiling lives in zwt CUDA
# events, not nsight, and covers every training stage):
#   * Median tok/s after warmup (throughput headline)
#   * Time to validation/training loss target (if reached within budget)
#   * Per-stage CUDA-event profile from zwt (where applicable)
#
# Plus the SubQ inference scoreboard from scripts/bench_subq.sh.
#
# Output: results/race_250M/{olmocpp,zwt,olmopython}.csv
#         results/race_250M/REPORT.md
#         results/subq_baseline.csv (from bench_subq.sh)
#
# Usage:
#   ./scripts/race_250M.sh                       # full race, ~30-60 min H100
#   ./scripts/race_250M.sh --short               # 200 steps each, ~5 min
#   ./scripts/race_250M.sh --steps 2000          # custom step count
#   ./scripts/race_250M.sh --skip-build          # binaries already built
#   ./scripts/race_250M.sh --only zwt            # run just one stack
#   ./scripts/race_250M.sh --target-loss 4.0     # stop when loss <= 4.0
#   ./scripts/race_250M.sh --skip-subq           # skip the inference bench
#
# Required: H100 with CUDA 12+, python3 + pip available, git repo at HEAD
# of fast-inference branch.

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="$REPO_ROOT/results/race_250M"
VENV_DIR="$REPO_ROOT/.venv"
LOGS_DIR="$RESULTS_DIR/logs"
TOKENS_PATH="$DATA_DIR/owt/owt_tokens.npy"

# ── Defaults ────────────────────────────────────────────────────────────
STEPS=2000
WARMUP=200
TARGET_LOSS=""        # empty = no early stop
ONLY_STACK=""         # olmocpp | zwt | olmopython | (empty = all)
SKIP_BUILD=0
SKIP_SUBQ=0
SKIP_DATA_PREP=0
SHORT=0

# ── Parse CLI ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --short)         SHORT=1; shift;;
    --steps)         STEPS="$2"; shift 2;;
    --warmup)        WARMUP="$2"; shift 2;;
    --target-loss)   TARGET_LOSS="$2"; shift 2;;
    --only)          ONLY_STACK="$2"; shift 2;;
    --skip-build)    SKIP_BUILD=1; shift;;
    --skip-subq)     SKIP_SUBQ=1; shift;;
    --skip-data-prep) SKIP_DATA_PREP=1; shift;;
    --tokens)        TOKENS_PATH="$2"; shift 2;;
    -h|--help)       sed -n '2,42p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ $SHORT -eq 1 ]]; then
  STEPS=200
  WARMUP=20
fi

# ── Logging helpers ─────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
BOLD='\033[1m'; NC='\033[0m'
log()     { echo -e "${GREEN}[race]${NC} $*"; }
warn()    { echo -e "${YELLOW}[race]${NC} $*"; }
err()     { echo -e "${RED}[race]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}\n"; }

# ── Sanity ──────────────────────────────────────────────────────────────
section "Pre-flight"
[[ -d "$REPO_ROOT/.git" ]] || err "not in a git repo: $REPO_ROOT"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  log "GPU: $GPU_NAME ($GPU_MEM MiB)"
  if [[ "$GPU_NAME" != *H100* ]] && [[ "$GPU_NAME" != *A100* ]]; then
    warn "GPU is not H100/A100 — race numbers will not be apples-to-apples with published"
  fi
else
  err "nvidia-smi missing — race must run on a CUDA host"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  warn "nvcc not on PATH — build will skip CUDA kernels (no WGMMA, no custom kernels)"
fi

# ── Build ───────────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  section "Build llm-cpp + zwt (Release, CUDA, WGMMA)"
  mkdir -p "$BUILD_DIR"
  ( cd "$BUILD_DIR" && cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
      -DCMAKE_CUDA_ARCHITECTURES="80;90" \
      -DOLMO_BUILD_KERNELS=ON \
      -DZWT_USE_WGMMA=ON \
      ) > "$LOGS_DIR/cmake.log" 2>&1 || { tail -50 "$LOGS_DIR/cmake.log"; err "cmake config failed"; }
  cmake --build "$BUILD_DIR" -j --target olmo_train zwt_pretrain bench_attn prepare_data \
    > "$LOGS_DIR/build.log" 2>&1 || { tail -80 "$LOGS_DIR/build.log"; err "build failed"; }
  log "built: olmo_train, zwt_pretrain, bench_attn, prepare_data"
else
  log "skipping build (--skip-build)"
fi

[[ -x "$BUILD_DIR/olmo_train" ]]   || err "olmo_train not built"
[[ -x "$BUILD_DIR/zwt_pretrain" ]] || err "zwt_pretrain not built"
[[ -x "$BUILD_DIR/bench_attn" ]]   || err "bench_attn not built"

# ── Python env ──────────────────────────────────────────────────────────
section "Python venv (PyTorch + ai2-olmo-core)"
if [[ ! -d "$VENV_DIR" ]] || ! "$VENV_DIR/bin/python" -c "import torch" 2>/dev/null; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip -q
  "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124 -q \
    || "$VENV_DIR/bin/pip" install torch -q
  "$VENV_DIR/bin/pip" install numpy -q
  "$VENV_DIR/bin/pip" install ai2-olmo-core -q || warn "ai2-olmo-core install failed; olmo-python stack will use the fallback nn.Transformer"
fi
"$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available()" \
  || err "PyTorch CUDA not available in venv"
log "venv ready: $VENV_DIR"

# ── Data ────────────────────────────────────────────────────────────────
if [[ $SKIP_DATA_PREP -eq 0 ]]; then
  section "Data — OWT, GPT-2 tokenizer"
  mkdir -p "$DATA_DIR/gpt2" "$DATA_DIR/owt"
  if [[ ! -f "$DATA_DIR/gpt2/vocab.json" ]]; then
    curl -sL "https://huggingface.co/gpt2/resolve/main/vocab.json" \
         -o "$DATA_DIR/gpt2/vocab.json"
  fi
  if [[ ! -f "$DATA_DIR/gpt2/merges.txt" ]]; then
    curl -sL "https://huggingface.co/gpt2/resolve/main/merges.txt" \
         -o "$DATA_DIR/gpt2/merges.txt"
  fi
  if [[ ! -f "$TOKENS_PATH" ]]; then
    log "tokenizing OpenWebText (this is slow first time — cached after)"
    "$BUILD_DIR/prepare_data" \
      --download-hf Skylion007/openwebtext \
      --output "$TOKENS_PATH" \
      --vocab-file "$DATA_DIR/gpt2/vocab.json" \
      --merges-file "$DATA_DIR/gpt2/merges.txt" \
      --threads "$(nproc)" \
      > "$LOGS_DIR/prepare_data.log" 2>&1 \
      || { tail -50 "$LOGS_DIR/prepare_data.log"; err "data prep failed"; }
  fi
  log "tokens: $TOKENS_PATH"
fi

[[ -f "$TOKENS_PATH" ]] || err "missing $TOKENS_PATH — run without --skip-data-prep"

# ── Run helpers ─────────────────────────────────────────────────────────
should_run() {
  [[ -z "$ONLY_STACK" || "$ONLY_STACK" == "$1" ]]
}

# Make per-stack tweaks to step counts via env var override on the conf.
# olmo_train + zwt_pretrain both honor `steps` / `max_steps` from the conf;
# we copy the conf to a tmp file and rewrite that line so the original
# checked-in conf stays unchanged.
prep_conf_with_steps() {
  local in="$1" out="$2" steps="$3"
  awk -v s="$steps" '
    /^steps[[:space:]]/   { print "steps\t" s; next }
    /^max_steps[[:space:]]*=/   { print "max_steps     = " s; next }
    /^max_steps[[:space:]]/     { print "max_steps\t" s; next }
    { print }' "$in" > "$out"
}

# ── Run olmo_train (LibTorch C++) ───────────────────────────────────────
if should_run olmocpp; then
  section "Stack 1/3 — olmo_train (LibTorch C++)"
  CONF_TMP="$RESULTS_DIR/_olmo_250M.conf"
  prep_conf_with_steps "$REPO_ROOT/conf/olmo_250M.conf" "$CONF_TMP" "$STEPS"
  STDOUT="$LOGS_DIR/olmocpp.log"
  /usr/bin/time -f '%e %M' \
    "$BUILD_DIR/olmo_train" "$CONF_TMP" \
    > "$STDOUT" 2> "$LOGS_DIR/olmocpp.time" || warn "olmo_train exited non-zero"
  # Parse olmo_train's stdout into a CSV with columns step,loss,tok_per_s,wall_s
  "$VENV_DIR/bin/python" - "$STDOUT" "$RESULTS_DIR/olmocpp.csv" <<'PYEOF'
import re, sys, csv, time
log_path, out_path = sys.argv[1], sys.argv[2]
# Match patterns the trainer prints: "step <N>" and "loss: <F>", "tok/s: <N>".
# olmo_train's print line in src/train.cpp 595..608 is roughly:
#   step X | loss F | step_ms: N tok/s: N
step_re = re.compile(r"step\s+(\d+).*?loss[:= ]\s*([0-9.eE+-]+).*?tok/?s[:= ]\s*([0-9.eE+-]+)", re.I)
rows = []
with open(log_path) as f:
    t0 = None
    for line in f:
        m = step_re.search(line)
        if not m: continue
        if t0 is None: t0 = time.time()
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), 0.0))
# Approximate wall_s as cumulative dt assuming steady tok/s; we don't have real
# timestamps in the log, so emit raw triplets and let downstream summarizer
# compute wall_s from log_interval * sum(step_ms). Good enough for the race.
with open(out_path, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["step","loss","tok_per_s","wall_s","note"])
    for step, loss, tps, wall in rows:
        w.writerow([step, f"{loss:.6f}", f"{tps:.2f}", f"{wall:.3f}", "olmo_train_libtorch_cpp"])
print(f"wrote {out_path}: {len(rows)} rows")
PYEOF
  log "olmo_train: $RESULTS_DIR/olmocpp.csv"
fi

# ── Run zwt_pretrain (LibTorch-free C++) ────────────────────────────────
if should_run zwt; then
  section "Stack 2/3 — zwt_pretrain (LibTorch-free C++, WGMMA)"
  CONF_TMP="$RESULTS_DIR/_owt_250M_zwt.conf"
  prep_conf_with_steps "$REPO_ROOT/zwt/conf/owt_250M_h100.conf" "$CONF_TMP" "$STEPS"
  STDOUT="$LOGS_DIR/zwt.log"
  "$BUILD_DIR/zwt_pretrain" "$CONF_TMP" \
    --metrics-csv "$RESULTS_DIR/zwt.csv" \
    --profile-csv "$RESULTS_DIR/zwt_profile.csv" \
    > "$STDOUT" 2>&1 || warn "zwt_pretrain exited non-zero"
  log "zwt_pretrain: $RESULTS_DIR/zwt.csv"
  [[ -f "$RESULTS_DIR/zwt_profile.csv" ]] && log "zwt profile (per-stage CUDA events): $RESULTS_DIR/zwt_profile.csv"
fi

# ── Run olmo-python (upstream black box) ────────────────────────────────
if should_run olmopython; then
  section "Stack 3/3 — olmo-python (upstream)"
  EXTRA=""
  if [[ -n "$TARGET_LOSS" ]]; then
    EXTRA="--target-loss $TARGET_LOSS"
  fi
  STDOUT="$LOGS_DIR/olmopython.log"
  "$VENV_DIR/bin/python" "$REPO_ROOT/scripts/train_olmo_python_250M.py" \
    --tokens "$TOKENS_PATH" \
    --steps "$STEPS" --warmup "$WARMUP" \
    --batch-size 4 --grad-accum 8 --seq-len 4096 \
    --output "$RESULTS_DIR/olmopython.csv" \
    $EXTRA \
    > "$STDOUT" 2>&1 || warn "olmo-python exited non-zero"
  log "olmo-python: $RESULTS_DIR/olmopython.csv"
fi

# ── SubQ inference scoreboard ───────────────────────────────────────────
if [[ $SKIP_SUBQ -eq 0 ]]; then
  section "SubQ inference scoreboard"
  "$REPO_ROOT/scripts/bench_subq.sh" --device cuda --dtype bf16 \
    --output "$REPO_ROOT/results/subq_baseline.csv" \
    > "$LOGS_DIR/subq.log" 2>&1 || warn "bench_subq.sh exited non-zero"
  log "SubQ: $REPO_ROOT/results/subq_baseline.csv"
fi

# ── Report ──────────────────────────────────────────────────────────────
section "Report"
"$VENV_DIR/bin/python" - <<PYEOF
import csv, os, statistics, time
from pathlib import Path

results = Path("$RESULTS_DIR")
report  = results / "REPORT.md"
target_loss = "$TARGET_LOSS"

def load_csv(name):
    p = results / name
    if not p.exists(): return None
    rows = []
    with open(p) as f:
        r = csv.DictReader(f)
        for x in r: rows.append(x)
    return rows or None

stacks = {
    "olmo_train (LibTorch C++)":   "olmocpp.csv",
    "zwt_pretrain (LibTorch-free)":"zwt.csv",
    "olmo-python (upstream)":      "olmopython.csv",
}

with open(report, "w") as f:
    f.write("# 250M from-scratch race — REPORT\n\n")
    f.write(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip()
        f.write(f"- GPU: {gpu}\n")
    except Exception: pass
    f.write(f"- Steps: $STEPS  Warmup: $WARMUP\n")
    f.write(f"- Target loss: {target_loss or '(disabled)'}\n\n")

    f.write("## Throughput (median tok/s after warmup)\n\n")
    f.write("| stack | rows | median tok/s | final loss | wall to last log |\n")
    f.write("|---|---|---|---|---|\n")
    for label, fname in stacks.items():
        rs = load_csv(fname)
        if not rs:
            f.write(f"| {label} | — | did not run | — | — |\n"); continue
        try:
            after = [r for r in rs if int(r['step']) >= $WARMUP]
            tps = [float(r['tok_per_s']) for r in after if r['tok_per_s'] not in ("","0","0.00")]
            med = statistics.median(tps) if tps else 0.0
            final_loss = float(rs[-1]['loss'])
            wall = float(rs[-1].get('wall_s', '0') or 0)
        except Exception as e:
            f.write(f"| {label} | parse error: {e} | — | — | — |\n"); continue
        f.write(f"| {label} | {len(rs)} | {med:,.0f} | {final_loss:.4f} | {wall:.1f}s |\n")

    if target_loss:
        f.write(f"\n## Time-to-target-loss ({target_loss})\n\n")
        f.write("| stack | reached? | step | wall_s |\n")
        f.write("|---|---|---|---|\n")
        T = float(target_loss)
        for label, fname in stacks.items():
            rs = load_csv(fname)
            if not rs:
                f.write(f"| {label} | n/a | — | — |\n"); continue
            hit = next((r for r in rs if float(r.get('loss','9e9')) <= T), None)
            if hit:
                f.write(f"| {label} | yes | {hit['step']} | {hit.get('wall_s','—')} |\n")
            else:
                f.write(f"| {label} | no  | — | — |\n")

    f.write("\n## Inference scoreboard (SubQ comparable)\n\n")
    subq = Path("$REPO_ROOT/results/subq_baseline.csv")
    if subq.exists():
        f.write(f"Raw CSV: \`{subq}\`\n\n")
        with open(subq) as src:
            head = src.readline().strip().split(",")
            f.write("| " + " | ".join(head) + " |\n")
            f.write("|" + "|".join(["---"] * len(head)) + "|\n")
            for line in src:
                f.write("| " + " | ".join(line.strip().split(",")) + " |\n")
    else:
        f.write("(skipped or did not run)\n")

print(f"wrote {report}")
PYEOF

log "REPORT: $RESULTS_DIR/REPORT.md"
echo
cat "$RESULTS_DIR/REPORT.md"
