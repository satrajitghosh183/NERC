#!/usr/bin/env bash
# scripts/launch_multigpu.sh — single-node multi-GPU training (NCCL DDP).
#
# One olmo_train process per GPU. Each rank is isolated to its GPU via
# CUDA_VISIBLE_DEVICES (so all in-process placement is cuda:0 and NCCL handles
# the cross-GPU collectives). DDP replicates the model and averages gradients;
# the data loader shards by rank (set_shard) so ranks see disjoint data.
#
# REQUIREMENTS:
#   - Build with NCCL:   ./scripts/build.sh --cuda --nccl
#   - The conf must use  gpu_data = 0   (the GPU-resident path is not yet
#     rank-sharded; CPU streaming is). cuda_graph auto-disables under DDP.
#   - Model must FIT ON ONE GPU (this is DDP = full replication). For a model
#     too big for one GPU (3.6B), that's FSDP — Rung 2, not this.
#   - Data + save paths in the conf must be valid ON THIS box.
#
# Usage:
#   ./scripts/launch_multigpu.sh conf/olmo_1B_h100.conf        # use all GPUs
#   ./scripts/launch_multigpu.sh conf/olmo_1B_h100.conf 2      # force 2 GPUs
#   MASTER_PORT=29502 ./scripts/launch_multigpu.sh <conf> 2    # if port busy

set -uo pipefail
cd "$(dirname "$0")/.."

CONF="${1:?usage: launch_multigpu.sh <conf> [num_gpus]}"
NGPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
BUILD_DIR="${BUILD_DIR:-build}"
PORT="${MASTER_PORT:-29501}"
LOGDIR="${LOGDIR:-runs/multigpu}"
mkdir -p "$LOGDIR"

[[ -x "$BUILD_DIR/olmo_train" ]] || { echo "ERROR: $BUILD_DIR/olmo_train missing — ./scripts/build.sh --cuda --nccl"; exit 1; }
[[ -f "$CONF" ]] || { echo "ERROR: conf not found: $CONF"; exit 1; }
[[ "$NGPU" -ge 2 ]] || { echo "ERROR: need >=2 GPUs (found $NGPU). For 1 GPU use ./build/olmo_train $CONF"; exit 1; }

# Sanity: warn if the conf uses GPU-resident data (not rank-sharded yet).
if grep -qE '^[[:space:]]*gpu_data[[:space:]]*[=:]?[[:space:]]*1' "$CONF"; then
  echo "WARN: $CONF has gpu_data=1 — the GPU-resident loader is NOT rank-sharded;"
  echo "      set gpu_data=0 so each rank streams a disjoint data slice."
fi

echo "== Multi-GPU DDP: $NGPU ranks · $CONF · NCCL · port $PORT =="
pids=()
for ((r=0; r<NGPU; r++)); do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD_SIZE=$NGPU LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
    "$BUILD_DIR/olmo_train" "$CONF" > "$LOGDIR/rank_$r.log" 2>&1 &
  pids+=($!)
  echo "  rank $r -> GPU $r  (pid ${pids[-1]}, log $LOGDIR/rank_$r.log)"
done

# Stream rank 0 live; clean up everything on exit/interrupt.
tail -f "$LOGDIR/rank_0.log" & TAIL=$!
trap 'kill "${pids[@]}" "$TAIL" 2>/dev/null' EXIT INT TERM

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
kill "$TAIL" 2>/dev/null || true
if [[ $rc -ne 0 ]]; then
  echo "ERROR: a rank failed — inspect $LOGDIR/rank_*.log (rank 0 streamed above)."
  exit 1
fi
echo "== all $NGPU ranks finished OK =="
