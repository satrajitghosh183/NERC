#!/usr/bin/env bash
# scripts/race/07_infer_python.sh
#
# Python inference race. Loads the Python-trained 250M checkpoint and
# generates N tokens from the same fixed prompt the C++ side uses,
# measured the same way (median tok/s over K trials).
#
# This is the comparison the user wants: vanilla PyTorch generation
# (no MTP, no paged KV, no speculative). The C++ side's MTP-speculative
# decode should beat this on tok/s.
#
# Uses scripts/race/python_inference_baseline.py — a minimal but
# correct PyTorch generation loop. Replace with vllm/HF generate by
# setting PY_INFER_CMD.

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/python_infer
mkdir -p "$results_dir"

say() { printf "\033[1;36m[infer-py]\033[0m %s\n" "$*"; }

CKPT="${PY_CKPT:-scripts/race/results/python_ckpt/last.pt}"
PROMPT="Once upon a time in a small village by the mountain, there lived"
TRIALS=5
TOKENS=256

say "ckpt:   $CKPT"
say "prompt: $PROMPT"
say "trials: $TRIALS × $TOKENS tokens"

LOG="$results_dir/infer.log"
RESULTS_CSV="$results_dir/results.csv"
echo "trial,tok_per_s,total_tokens,wall_seconds" > "$RESULTS_CSV"

INFER_CMD="${PY_INFER_CMD:-python3 scripts/race/python_inference_baseline.py}"

for trial in $(seq 1 "$TRIALS"); do
  say "trial $trial/$TRIALS …"
  trial_log="$results_dir/trial_${trial}.log"
  $INFER_CMD \
      --checkpoint "$CKPT" \
      --tokenizer-vocab data/gpt2/vocab.json \
      --tokenizer-merges data/gpt2/merges.txt \
      --prompt "$PROMPT" \
      --max-new-tokens "$TOKENS" \
      --device cuda \
      2>&1 | tee "$trial_log"

  python3 - <<EOF
import re
with open("$trial_log") as f:
    text = f.read()
m = re.search(r"\[(\d+)\s+tokens,\s+([\d\.]+)\s+tok/s.*?\]", text)
if m:
    tokens, toks = int(m.group(1)), float(m.group(2))
    wall = tokens / toks if toks > 0 else 0.0
    with open("$RESULTS_CSV", "a") as out:
        out.write(f"$trial,{toks},{tokens},{wall:.3f}\n")
    print(f"  → {toks:.1f} tok/s")
EOF
done

python3 - <<EOF
import csv, statistics
rows = list(csv.DictReader(open("$RESULTS_CSV")))
if not rows:
    print("  no successful trials")
else:
    speeds = [float(r["tok_per_s"]) for r in rows]
    print(f"\n  median tok/s: {statistics.median(speeds):.1f}")
    print(f"  mean   tok/s: {statistics.mean(speeds):.1f}")
    print(f"  min    tok/s: {min(speeds):.1f}")
    print(f"  max    tok/s: {max(speeds):.1f}")
EOF
