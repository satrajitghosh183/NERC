#!/usr/bin/env bash
# scripts/race/06_infer_cpp.sh
#
# C++ inference race. Uses the chat tool's MTP speculative decoding +
# paged KV cache to generate N tokens from a fixed prompt. Reports
# median tok/s over K trials.
#
# This is where the MTP heads pay off: the linear-chain speculative
# decoder drafts via MTP and verifies, accepting 2-4 tokens per
# verify forward instead of 1.

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/cpp_infer
mkdir -p "$results_dir"

say() { printf "\033[1;36m[infer-cpp]\033[0m %s\n" "$*"; }

BUILD_DIR="${BUILD_DIR:-build}"
CKPT="${CPP_CKPT:-scripts/race/results/cpp_ckpt/model.pt}"
CONF=scripts/race/configs/race_250m_cpp.json
VOCAB=data/gpt2/vocab.json
MERGES=data/gpt2/merges.txt

if [[ ! -f "$CKPT" ]]; then
  say "ERROR: no C++ checkpoint at $CKPT"
  say "       run phase 04 (race_train_cpp) first, or set CPP_CKPT=<path>"
  say "       NOTE: do NOT use checkpoints/125M.pt — wrong architecture"
  say "       (125M is d_model=768/vocab=50257; race conf needs d_model=1024/vocab=50304)"
  exit 1
fi

PROMPT="Once upon a time in a small village by the mountain, there lived"
TRIALS=5
TOKENS=256

say "ckpt:   $CKPT"
say "prompt: $PROMPT"
say "trials: $TRIALS × $TOKENS tokens"

# Reject stale/wrong checkpoints before launching chat (e.g. old TorchScript
# 125M/768d exports vs race_250m_cpp 1024d C++ torch::save output).
python3 - <<EOF
import json, sys, torch
from pathlib import Path

ckpt = Path("$CKPT")
conf = Path("$CONF")
cfg = json.load(open(conf))

# BEST-EFFORT pre-check ONLY — never blocks. The C++ checkpoint is a libtorch
# (TorchScript-style) archive: Python's torch.load can't reliably parse it as a
# plain state_dict, and PyTorch 2.6 made weights_only=True the default, which
# rejects such archives outright. The authoritative loader is the C++ `chat`
# binary (it does its own arch alignment + load). So on ANY problem here we WARN
# and proceed; chat will load the phase-04 checkpoint and fail clearly if wrong.
try:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"[infer-cpp] note: skipping Python pre-check ({type(e).__name__}); chat will load + validate", file=sys.stderr)
    sys.exit(0)

sd = obj.get("state_dict", obj) if isinstance(obj, dict) else None
if not isinstance(sd, dict):
    print("[infer-cpp] note: C++ checkpoint not introspectable from Python; chat will load + validate", file=sys.stderr)
    sys.exit(0)

key = next(
    (k for k in sd
     if ("embed" in k.lower() or "wte" in k.lower())
     and isinstance(sd[k], torch.Tensor)
     and sd[k].ndim == 2),
    None,
)
if key is None:
    print("[infer-cpp] note: no embedding weight visible from Python; chat will validate", file=sys.stderr)
    sys.exit(0)

vocab, d_model = sd[key].shape
exp_v, exp_d = cfg["vocab_size"], cfg["d_model"]
if vocab != exp_v or d_model != exp_d:
    print(f"[infer-cpp] WARN: checkpoint embed [{vocab}, {d_model}] != config [{exp_v}, {exp_d}]", file=sys.stderr)
    if d_model == 768 and vocab in (50257, 50304):
        print("[infer-cpp]       looks like the 125M checkpoint — wrong model for this race", file=sys.stderr)
    sys.exit(0)   # still let chat be the judge

print(f"[infer-cpp] checkpoint OK: d_model={d_model}, vocab={vocab}", flush=True)
EOF

LOG="$results_dir/infer.log"
RESULTS_CSV="$results_dir/results.csv"
echo "trial,tok_per_s,total_tokens,wall_seconds,accept_rate" > "$RESULTS_CSV"

# Five fresh runs (each restarts the chat process so state is clean).
for trial in $(seq 1 "$TRIALS"); do
  say "trial $trial/$TRIALS …"
  trial_log="$results_dir/trial_${trial}.log"
  echo "$PROMPT" | "$BUILD_DIR/chat" \
      --checkpoint "$CKPT" \
      --config "$CONF" \
      --vocab-file "$VOCAB" \
      --merges-file "$MERGES" \
      --max-tokens "$TOKENS" \
      --temperature 0 \
      --paged-kv \
      --device cuda \
      2>&1 | tee "$trial_log"

  # Extract from the chat tool's trailing stats line:
  #   [256 tokens, 187.4 tok/s, speculative, 64% accepted]
  python3 - <<EOF
import re
with open("$trial_log") as f:
    text = f.read()
m = re.search(r"\[(\d+)\s+tokens,\s+([\d\.]+)\s+tok/s.*?(\d+)%\s+accepted\]", text)
if m:
    tokens, toks, accept = int(m.group(1)), float(m.group(2)), int(m.group(3))
    wall = tokens / toks if toks > 0 else 0.0
    with open("$RESULTS_CSV", "a") as out:
        out.write(f"$trial,{toks},{tokens},{wall:.3f},{accept}\n")
    print(f"  → {toks:.1f} tok/s, accept {accept}%")
else:
    print(f"  (no stats line found in trial $trial — check $trial_log)")
EOF
done

# Summary
python3 - <<EOF
import csv, statistics
rows = list(csv.DictReader(open("$RESULTS_CSV")))
if not rows:
    print("  no successful trials")
else:
    speeds = [float(r["tok_per_s"]) for r in rows]
    accepts = [float(r["accept_rate"]) for r in rows]
    print(f"\n  median tok/s: {statistics.median(speeds):.1f}")
    print(f"  mean   tok/s: {statistics.mean(speeds):.1f}")
    print(f"  min    tok/s: {min(speeds):.1f}")
    print(f"  max    tok/s: {max(speeds):.1f}")
    print(f"  median accept_rate: {statistics.median(accepts):.0f}%")
EOF
