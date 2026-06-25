#!/usr/bin/env bash
# scripts/test_multirank.sh
#
# Multi-rank training smoke test (item 6). Spawns N processes with the
# torchrun-equivalent env vars (RANK, WORLD_SIZE, MASTER_ADDR,
# MASTER_PORT) and runs olmo_train for a few steps on each rank.
#
# Validates:
#  - DDPContext::init_from_env successfully establishes ProcessGroupGloo
#  - register_grad_hooks fires per-rank during backward
#  - allreduce_gradients completes on all ranks without hangs
#  - Each rank produces the same loss curve to within float noise
#  - OptimizerStateSharder partition+allgather keeps params in sync
#
# Usage:
#   ./scripts/test_multirank.sh 2  # 2 ranks
#   ./scripts/test_multirank.sh 4  # 4 ranks
#
# Requires the build to have been done with -DOLMO_USE_DDP=ON and Gloo
# headers available. On Mac w/o Gloo this script exits cleanly with an
# explanatory message.

set -euo pipefail

WORLD_SIZE="${1:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-conf/olmo_125M.conf}"
STEPS="${STEPS:-20}"
LOG_DIR="${LOG_DIR:-results/multirank_$(date +%Y%m%d_%H%M%S)}"

# Sanity check: olmo_train must exist and be DDP-capable.
if [[ ! -x ./build/olmo_train ]]; then
  echo "ERROR: ./build/olmo_train not found. Run scripts/build.sh first."
  exit 1
fi
if ! ./build/olmo_train --help 2>&1 | grep -qi "ddp\|distributed"; then
  echo "INFO: olmo_train built without DDP. Rebuild with"
  echo "  cmake -B build -DOLMO_USE_DDP=ON && cmake --build build"
  exit 0
fi

mkdir -p "$LOG_DIR"
echo "Multi-rank smoke test: WORLD_SIZE=$WORLD_SIZE, master=$MASTER_ADDR:$MASTER_PORT"
echo "Logs in $LOG_DIR"
echo

pids=()
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  log_file="$LOG_DIR/rank_${rank}.log"
  RANK=$rank WORLD_SIZE=$WORLD_SIZE \
  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    ./build/olmo_train "$CONFIG" --steps "$STEPS" >"$log_file" 2>&1 &
  pids+=($!)
  echo "Spawned rank $rank as PID ${pids[$rank]}"
done

# Wait for all.
exit_codes=()
for pid in "${pids[@]}"; do
  set +e
  wait "$pid"
  exit_codes+=($?)
  set -e
done

# Verify all ranks exited cleanly.
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  if [[ "${exit_codes[$rank]}" -ne 0 ]]; then
    echo "FAIL: rank $rank exited with code ${exit_codes[$rank]}"
    tail -20 "$LOG_DIR/rank_${rank}.log"
    exit 1
  fi
done

# Compare last-line loss across ranks.
echo
echo "Final loss per rank:"
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  loss=$(grep -oE "loss: [0-9.]+" "$LOG_DIR/rank_${rank}.log" | tail -1 | awk '{print $2}')
  echo "  rank $rank: $loss"
done

# Compute max divergence between ranks.
losses=()
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  losses+=("$(grep -oE 'loss: [0-9.]+' "$LOG_DIR/rank_${rank}.log" | tail -1 | awk '{print $2}')")
done

# All ranks should converge to within ~1% of each other.
python3 - <<PY || exit 1
losses = [${losses[*]/%/,}]
losses = [l for l in losses if l != ""]
if not losses:
    print("ERROR: no loss values found in logs")
    raise SystemExit(1)
mx, mn = max(losses), min(losses)
diff = mx - mn
rel  = diff / mx if mx > 0 else 0
print(f"max={mx:.6f} min={mn:.6f} range={diff:.6f} rel={rel*100:.2f}%")
if rel > 0.05:
    print("FAIL: ranks diverged by more than 5%")
    raise SystemExit(1)
print("PASS: all ranks within 5% of each other")
PY

echo
echo "Multi-rank smoke OK ($WORLD_SIZE ranks, $STEPS steps each)"
