#!/usr/bin/env bash
# launch_ddp.sh — spawn N copies of zwt_pretrain, one per local GPU rank.
#
# Each child gets:
#   RANK=$R, LOCAL_RANK=$R, WORLD_SIZE=$WORLD,
#   MASTER_ADDR=$ADDR, MASTER_PORT=$PORT,
#   CUDA_VISIBLE_DEVICES is intentionally NOT set — zwt_pretrain calls
#   cudaSetDevice(local_rank) directly, which is the right pattern when
#   you're sharing a process group's NVLink fabric. Override by exporting
#   CUDA_VISIBLE_DEVICES before invoking this script if you want to mask
#   GPUs out (e.g. on a 4-GPU box where you only want to use 0+1).
#
# Logs go to logs/zwt_ddp_r<R>.log; rank 0's log is also tee'd to stdout
# so you can `tail -f` on a fresh shell. Exit code is rank 0's exit code
# (which usually propagates the first failure).
#
# Usage:
#   bash zwt/scripts/launch_ddp.sh <config.conf>             # defaults: world=2, port=29500
#   bash zwt/scripts/launch_ddp.sh <config> 4                # 4 local GPUs
#   bash zwt/scripts/launch_ddp.sh <config> 2 30000          # custom port
#   MASTER_ADDR=10.0.0.1 bash zwt/scripts/launch_ddp.sh <conf> 2   # override addr

set -euo pipefail

CONFIG=${1:?usage: $0 <config.conf> [world_size=2] [master_port=29500]}
WORLD=${2:-2}
PORT=${3:-29500}
ADDR=${MASTER_ADDR:-127.0.0.1}
BIN=${ZWT_BIN:-./build/zwt_pretrain}
EXTRA_ARGS=()
# Pass any additional args after the third positional through to zwt_pretrain.
if [[ $# -gt 3 ]]; then
  shift 3
  EXTRA_ARGS=("$@")
fi

if [[ ! -x "$BIN" ]]; then
  echo "launch_ddp: $BIN not executable — build first: ./scripts/build.sh --cuda" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "launch_ddp: $CONFIG missing" >&2
  exit 2
fi

mkdir -p logs

PIDS=()
cleanup() {
  # SIGTERM then a short grace, then SIGKILL anything still alive. Prevents
  # NCCL collectives from hanging the process group when one rank dies.
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "launch_ddp: world=$WORLD master=$ADDR:$PORT bin=$BIN config=$CONFIG"
echo "launch_ddp: extra args: ${EXTRA_ARGS[*]:-(none)}"

for ((R=0; R<WORLD; R++)); do
  LOG="logs/zwt_ddp_r${R}.log"
  if [[ "$R" -eq 0 ]]; then
    # Rank 0's stderr also flows to the launching shell so the user sees
    # progress without `tail -f`.
    RANK=$R LOCAL_RANK=$R WORLD_SIZE=$WORLD \
      MASTER_ADDR=$ADDR MASTER_PORT=$PORT \
      stdbuf -oL -eL "$BIN" "$CONFIG" "${EXTRA_ARGS[@]}" 2> >(tee "$LOG" >&2) >/dev/null &
  else
    RANK=$R LOCAL_RANK=$R WORLD_SIZE=$WORLD \
      MASTER_ADDR=$ADDR MASTER_PORT=$PORT \
      "$BIN" "$CONFIG" "${EXTRA_ARGS[@]}" > "$LOG" 2>&1 &
  fi
  PIDS+=($!)
done

echo "launch_ddp: spawned ranks: ${PIDS[*]}"

# Wait for rank 0 specifically; if it dies, kill everyone (handled via trap).
wait "${PIDS[0]}"
RC0=$?

# Reap the rest. If they fail after rank 0 succeeded, surface the failure.
RC_REST=0
for ((R=1; R<WORLD; R++)); do
  if ! wait "${PIDS[$R]}"; then
    RC_REST=1
  fi
done

if [[ "$RC0" -ne 0 ]]; then
  exit "$RC0"
fi
exit "$RC_REST"
