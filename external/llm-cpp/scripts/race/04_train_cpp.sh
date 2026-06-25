#!/usr/bin/env bash
# scripts/race/04_train_cpp.sh
#
# Train the C++ 250M (backbone + MTP) for the step count specified in
# the conf. Captures step times, loss, and tok/s into a structured log.

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/cpp_train
mkdir -p "$results_dir"

say() { printf "\033[1;36m[cpp]\033[0m %s\n" "$*"; }

BUILD_DIR="${BUILD_DIR:-build}"

# Pick the config by available VRAM unless the user pins one with RACE_CONF.
# The default race conf is H100-sized (batch 32, seq 1024, gpu_data=1,
# cuda_graph=1) and OOMs below ~40 GB; the 5060ti conf is sized for 16 GB
# (batch 4 x grad_accum 8, gpu_data=0, cuda_graph=0, act-ckpt full).
if [[ -n "${RACE_CONF:-}" ]]; then
  CONF="$RACE_CONF"
else
  vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [[ -n "$vram_mb" && "$vram_mb" -lt 40000 ]]; then
    CONF=scripts/race/configs/race_250m_5060ti.conf
    say "detected ${vram_mb} MB VRAM (< 40 GB) → using 16 GB-safe config"
  else
    CONF=scripts/race/configs/race_250m_cpp.conf
    say "detected ${vram_mb:-unknown} MB VRAM → using H100 config"
  fi
fi
LOG="$results_dir/train.log"
METRICS="$results_dir/metrics.csv"

say "config: $CONF"
say "log:    $LOG"

# Wrap the train binary so we capture wall time + GPU utilization.
start=$(date +%s)
{
  echo "[run] $(date -u +%FT%TZ)"
  echo "[host] $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "[conf] $CONF"
  echo
  "$BUILD_DIR/olmo_train" "$CONF"
} 2>&1 | tee "$LOG"
end=$(date +%s)

say "wall clock: $((end - start)) s"

# Extract step-time + loss + tok/s from the log into a CSV the analyzer reads.
# olmo_train (src/train.cpp) prints lines like:
#   Epoch 0 | Step 100/1000  loss: 5.2341  lr: 3.00e-04  step_ms: 152  tok/s: 13456
python3 - <<EOF
import re, csv
rows = []
with open("$LOG") as f:
    for line in f:
        m = re.search(r"Step\s+(\d+)\b.*?loss:\s*([\d\.eE+-]+).*?step_ms:\s*(\d+).*?tok/s:\s*(\d+)", line)
        if m:
            rows.append({"step": int(m.group(1)),
                         "loss": float(m.group(2)),
                         "step_ms": float(m.group(3)),
                         "tok_per_s": float(m.group(4))})

with open("$METRICS", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step","loss","step_ms","tok_per_s"])
    w.writeheader()
    w.writerows(rows)

if rows:
    last = rows[-1]
    print(f"  {len(rows)} step records; final: step={last['step']} loss={last['loss']:.4f} "
          f"step_ms={last['step_ms']:.1f} tok/s={last['tok_per_s']:.0f}")
EOF
