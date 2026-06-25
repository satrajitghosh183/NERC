#!/usr/bin/env bash
# scripts/launch_big.sh — one command to train the 1B or 3B model with
# EVERYTHING (checkpoints, heartbeat, logs) on the attached volume, never the
# root disk. Runs inside tmux so you can disconnect and it keeps going.
#
# Usage:
#   ./scripts/launch_big.sh 1b      # ~1B  (AdamW, cuda_graph, single H100)
#   ./scripts/launch_big.sh 3b      # ~2.8B (AdamW + fp32 master, act-ckpt)
#
# tmux:
#   tmux attach -t big        # watch it
#   Ctrl+B then d             # detach (keeps running)
#   tmux kill-session -t big  # stop it
#
# Tunables via env:
#   BUILD_DIR (default build), VOL (default /media/volume/Prep_and_Voice_Training)

set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-}"
BUILD_DIR="${BUILD_DIR:-build}"
VOL="${VOL:-/media/volume/Prep_and_Voice_Training}"

case "$WHICH" in
  1b) CONF=conf/olmo_1B_h100.conf;       RUNDIR="$VOL/runs/1B" ;;
  3b) CONF=conf/olmo_3B_h100_adamw.conf; RUNDIR="$VOL/runs/3B" ;;
  *)  echo "Usage: $0 <1b|3b>"; exit 1 ;;
esac

say() { printf "\033[1;36m[launch]\033[0m %s\n" "$*"; }

# ── Preflight ──────────────────────────────────────────────────────────────
[[ -x "$BUILD_DIR/olmo_train" ]] || { echo "ERROR: $BUILD_DIR/olmo_train missing — run ./scripts/build.sh --cuda"; exit 1; }
[[ -d "$VOL" ]] || { echo "ERROR: volume not mounted at $VOL"; exit 1; }

DATA="$VOL/data/openwebtext_gpt2.npy"
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: training data not found at $DATA"
  echo "       point [data] data_path in $CONF at your tokenized .npy on the volume."
  exit 1
fi

mkdir -p "$RUNDIR"
LOG="$RUNDIR/train.log"

# Root-disk guard: refuse to start if root is dangerously full (would crash mid-run).
root_avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [[ -n "$root_avail_gb" && "$root_avail_gb" -lt 5 ]]; then
  echo "ERROR: only ${root_avail_gb} GB free on root disk — clean up first (see scripts/disk_cleanup.sh)."
  exit 1
fi

say "model:   $WHICH ($CONF)"
say "data:    $DATA"
say "outputs: $RUNDIR  (checkpoints + heartbeat + log, all on the volume)"
say "log:     $LOG"

# ── Launch in tmux ─────────────────────────────────────────────────────────
tmux kill-session -t big 2>/dev/null || true
tmux new-session -d -s big "stdbuf -oL -eL '$BUILD_DIR/olmo_train' '$CONF' 2>&1 | tee '$LOG'"

say "started in tmux session 'big'."
say "  watch:   tmux attach -t big      (detach: Ctrl+B then d)"
say "  tail:    tail -f $LOG"
say "  stop:    tmux kill-session -t big"
