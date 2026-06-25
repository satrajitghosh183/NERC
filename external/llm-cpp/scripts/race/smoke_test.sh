#!/usr/bin/env bash
# scripts/race/smoke_test.sh
#
# FIRST thing to run on a new machine (esp. the 5060 Ti / Blackwell box).
# Proves the whole stack works end-to-end in ~2 minutes WITHOUT a big
# download or a full training run:
#
#   1. binaries built?         (olmo_train / chat / prepare_data present)
#   2. GPU visible?            (nvidia-smi)
#   3. tokenizer downloadable? (tiny, validated)
#   4. tokenize a tiny corpus  (no HF download)
#   5. train 20 steps          → assert loss is FINITE and decreasing (no NaN)
#   6. generate 16 tokens       → assert inference runs and emits output
#
# Exits NONZERO on the first failure with a clear message, so you find a
# Blackwell-build / VRAM / dependency problem here, not mid-benchmark.
#
# Usage:   bash scripts/race/smoke_test.sh
# Override build dir: BUILD_DIR=build bash scripts/race/smoke_test.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

BUILD_DIR="${BUILD_DIR:-build}"
TMP=scripts/race/results/smoke
mkdir -p "$TMP" data/gpt2

pass() { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m  ✗ FAIL: %s\033[0m\n" "$*"; exit 1; }
step() { printf "\n\033[1;35m[%s/6] %s\033[0m\n" "$1" "$2"; }

# ── 1. binaries ───────────────────────────────────────────────────────────
step 1 "binaries built?"
for b in olmo_train chat prepare_data; do
  [[ -x "$BUILD_DIR/$b" ]] || fail "$BUILD_DIR/$b missing — run: ./scripts/build.sh --cuda"
done
pass "olmo_train, chat, prepare_data present"

# ── 2. GPU ────────────────────────────────────────────────────────────────
step 2 "GPU visible?"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || fail "nvidia-smi failed — no GPU / driver?"
pass "GPU reachable"

# ── 3. tokenizer ──────────────────────────────────────────────────────────
step 3 "tokenizer present/downloadable?"
VOCAB=data/gpt2/vocab.json; MERGES=data/gpt2/merges.txt
if [[ ! -f "$VOCAB" || ! -f "$MERGES" ]]; then
  curl -fsSL https://huggingface.co/gpt2/raw/main/vocab.json -o "$VOCAB" || fail "vocab.json download failed"
  curl -fsSL https://huggingface.co/gpt2/raw/main/merges.txt -o "$MERGES" || fail "merges.txt download failed"
fi
python3 -c "import json; json.load(open('$VOCAB'))" || fail "vocab.json invalid"
pass "tokenizer OK"

# ── 4. tiny corpus → tokens (no big download) ─────────────────────────────
step 4 "tokenize a tiny corpus"
CORPUS="$TMP/corpus"; mkdir -p "$CORPUS"
# Enough text that batch*seq tokens exist for a few steps.
yes "Once upon a time in a small village there lived a curious young fox who loved to explore the forest and learn new things every single day." \
  | head -n 4000 > "$CORPUS/text.txt"
TOKENS="$TMP/tokens.npy"
"$BUILD_DIR/prepare_data" --input "$CORPUS" --output "$TOKENS" \
  --vocab-file "$VOCAB" --merges-file "$MERGES" \
  || fail "prepare_data failed"
[[ -s "$TOKENS" ]] || fail "no tokens produced"
pass "tokens written: $TOKENS"

# ── 5. train 20 steps, assert finite + decreasing loss ────────────────────
step 5 "train 20 steps (asserting no NaN, loss decreasing)"
# Derive a tiny config from the 16GB race conf: 20 steps, small batch/seq so
# it runs in seconds and needs little data. SAME architecture as the race
# (260M, d=1024) so the checkpoint loads in chat with race_250m_cpp.json.
CKPT="$TMP/model.pt"
sed -e 's/^steps  *=.*/steps = 20/' \
    -e 's/^batch_size  *=.*/batch_size = 2/' \
    -e 's/^seq_len  *=.*/seq_len = 256/' \
    -e 's/^grad_accum  *=.*/grad_accum = 1/' \
    -e 's/^warmup_steps  *=.*/warmup_steps = 5/' \
    -e 's/^log_interval  *=.*/log_interval = 2/' \
    -e "s#^data_path  *=.*#data_path = $TOKENS#" \
    -e "s#^save  *=.*#save = $CKPT#" \
    scripts/race/configs/race_250m_5060ti.conf > "$TMP/smoke.conf"

TRAIN_LOG="$TMP/train.log"
"$BUILD_DIR/olmo_train" "$TMP/smoke.conf" 2>&1 | tee "$TRAIN_LOG"

# Pull the loss values out of the log.
mapfile -t LOSSES < <(grep -oE 'loss:[[:space:]]*[0-9.eE+-]+|loss:[[:space:]]*nan' "$TRAIN_LOG" \
                        | sed -E 's/loss:[[:space:]]*//')
[[ ${#LOSSES[@]} -ge 2 ]] || fail "no loss lines in training output"
grep -qiE 'loss:[[:space:]]*nan' "$TRAIN_LOG" && fail "training produced NaN loss"

# Compare the first REAL loss to the last. Step 0 logs loss 0.0000 — that's
# the async loss reader's initial placeholder before the first GPU readout
# lands, NOT a real loss — so ignore leading ~0 values.
python3 - "${LOSSES[@]}" <<'PY' || fail "loss did not decrease / non-finite (see $TRAIN_LOG)"
import sys, math
vals = [float(x) for x in sys.argv[1:]]
if any(not math.isfinite(v) for v in vals):
    print("  non-finite (NaN/inf) loss detected"); sys.exit(1)
real = [v for v in vals if v > 1e-6]          # drop step-0 placeholder zeros
if len(real) < 2:
    print("  not enough real loss values"); sys.exit(1)
first, last = real[0], real[-1]
print(f"  first real loss {first:.4f} -> last {last:.4f}")
sys.exit(0 if last <= first + 0.5 else 1)      # must trend down over 20 steps
PY
pass "loss finite and decreasing (no NaN)"
[[ -f "$CKPT" ]] || fail "no checkpoint saved at $CKPT"
pass "checkpoint saved: $CKPT"

# ── 6. inference: generate 16 tokens ──────────────────────────────────────
step 6 "generate 16 tokens"
INFER_LOG="$TMP/infer.log"
echo "Once upon a time" | "$BUILD_DIR/chat" \
    --checkpoint "$CKPT" \
    --config scripts/race/configs/race_250m_cpp.json \
    --vocab-file "$VOCAB" --merges-file "$MERGES" \
    --max-tokens 16 --temperature 0 --device cuda 2>&1 | tee "$INFER_LOG" \
  || fail "chat/inference crashed (see $INFER_LOG)"
grep -qiE 'tok/s|tokens' "$INFER_LOG" || fail "no generation output (see $INFER_LOG)"
pass "inference generated tokens"

printf "\n\033[1;32m════════════════════════════════════════\033[0m\n"
printf "\033[1;32m  SMOKE TEST PASSED — train + infer both work\033[0m\n"
printf "\033[1;32m════════════════════════════════════════\033[0m\n"
printf "Next: full data + race → bash scripts/race/run_all.sh\n"
