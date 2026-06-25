#!/usr/bin/env bash
# bench_threeway.sh — run zwt_pretrain, PyTorch eager, and torch.compile on
# the same machine with the same model shape and dump a comparison CSV.
#
# Pins:
#   * one process per stack, sequential (NVML jitter is enough to make
#     side-by-side numbers suspect)
#   * identical B, S, vocab, layers, heads, d_model, d_ffn
#   * identical warmup/iters
#   * CUDA_VISIBLE_DEVICES = 0 (override by exporting before invoking)
#
# Output: three CSV rows on stdout:
#   stack,dtype,ms_per_step,tok_per_sec,notes
#
# Usage:
#   zwt/scripts/bench_threeway.sh conf/owt_1B_h100.conf
#   zwt/scripts/bench_threeway.sh conf/owt_1B_h100.conf 100 10 2 512
#                                 config          iters warmup B S
#   zwt/scripts/bench_threeway.sh conf/owt_1B_2xh100.conf 50 5 "" "" 2
#                                                                 ↑ world
#
# When WORLD>1 each PT leg runs under torchrun --nproc-per-node=$WORLD and
# the zwt leg runs under launch_ddp.sh; the per-rank logs are stashed under
# logs/. tok/s already accounts for all ranks.
#
# The middle four args override iters/warmup/batch/seq for quick sanity runs.
set -euo pipefail

CONFIG=${1:?usage: $0 <config.conf> [iters warmup batch seq world]}
ITERS=${2:-50}
WARMUP=${3:-5}
BATCH=${4:-}
SEQ=${5:-}
WORLD=${6:-1}

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BIN_PT=$REPO/zwt/scripts/pt_baseline.py
BIN_OLMO=$REPO/zwt/scripts/run_olmo_python.py
BIN_ZWT=$REPO/build/zwt_pretrain

if [[ ! -x $BIN_ZWT ]]; then
  echo "bench_threeway: $BIN_ZWT not found; build with ./scripts/build.sh --cuda first" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "bench_threeway: nvidia-smi not on PATH — this benchmark needs CUDA" >&2
  exit 2
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "# bench_threeway on $GPU  config=$CONFIG  warmup=$WARMUP iters=$ITERS world=$WORLD" >&2

# zwt_pretrain reads data/owt/owt_tokens.npy from the .conf. Make sure the
# symlink is in place via find_tokens.sh — the long-run launchers do this
# but the bench script invokes the binary directly via launch_ddp.sh which
# does not. Skipped silently if find_tokens.sh fails (zwt leg will then
# error with a clear "TokenLoader: cannot open ..." which we tee through).
if TOKENS=$(bash "$REPO/zwt/scripts/find_tokens.sh" 2>/dev/null); then
  mkdir -p "$REPO/data/owt"
  ln -sfn "$TOKENS" "$REPO/data/owt/owt_tokens.npy"
  echo "# tokens: $TOKENS" >&2
fi

opt_batch=()
opt_seq=()
[[ -n $BATCH ]] && opt_batch=(--batch "$BATCH")
[[ -n $SEQ   ]] && opt_seq=(--seq "$SEQ")

# Choose how to launch the PyTorch leg based on WORLD. torchrun gives us
# RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT for free, which
# is what _ddp_init in pt_baseline.py reads.
pt_launch() {
  if [[ "$WORLD" -le 1 ]]; then
    python3 "$BIN_PT" "$@"
  else
    torchrun --nproc-per-node="$WORLD" --nnodes=1 --master-port=29501 \
             "$BIN_PT" "$@"
  fi
}

echo "stack,dtype,ms_per_step,tok_per_sec,notes"

# ---- PyTorch eager ----
PT_EAGER=$(pt_launch --config "$CONFIG" --warmup "$WARMUP" --steps "$ITERS" \
                   --dtype bf16 "${opt_batch[@]}" "${opt_seq[@]}" | tail -1)
# PT line format: "  B=.. S=.. ... dt=..s tok/s=... ms/step=.."
ms=$(echo "$PT_EAGER" | sed -n 's/.*ms\/step=\([0-9.]*\).*/\1/p')
tps=$(echo "$PT_EAGER" | sed -n 's/.*tok\/s=\([0-9,]*\).*/\1/p' | tr -d ,)
echo "pytorch_eager,bf16,$ms,$tps,world=$WORLD"

# ---- torch.compile ----
PT_COMP=$(pt_launch --config "$CONFIG" --warmup "$WARMUP" --steps "$ITERS" \
                  --dtype bf16 --compile "${opt_batch[@]}" "${opt_seq[@]}" | tail -1)
ms=$(echo "$PT_COMP"  | sed -n 's/.*ms\/step=\([0-9.]*\).*/\1/p')
tps=$(echo "$PT_COMP" | sed -n 's/.*tok\/s=\([0-9,]*\).*/\1/p' | tr -d ,)
echo "pytorch_compile,bf16,$ms,$tps,world=$WORLD"

# ---- olmo-python (AI2 OLMo-core's actual Transformer + AdamW + DDP) ----
# Same launch path as the PT legs (torchrun for WORLD>1) but invokes the
# olmo_core package directly. Skips silently if olmo_core isn't importable
# so single-host dev machines without `pip install -e ./olmo-python` still
# get a useful 3-row CSV.
if python3 -c 'import olmo_core' 2>/dev/null; then
  olmo_launch() {
    if [[ "$WORLD" -le 1 ]]; then
      python3 "$BIN_OLMO" "$@"
    else
      torchrun --nproc-per-node="$WORLD" --nnodes=1 --master-port=29503 \
               "$BIN_OLMO" "$@"
    fi
  }
  OLMO=$(olmo_launch --config "$CONFIG" --warmup "$WARMUP" --steps "$ITERS" \
                     --dtype bf16 "${opt_batch[@]}" "${opt_seq[@]}" | tail -1)
  ms=$(echo "$OLMO"  | sed -n 's/.*ms\/step=\([0-9.]*\).*/\1/p')
  tps=$(echo "$OLMO" | sed -n 's/.*tok\/s=\([0-9,]*\).*/\1/p' | tr -d ,)
  echo "olmo_python,bf16,$ms,$tps,world=$WORLD (AI2 OLMo-core, olmo2_1B_v2)"
else
  echo "olmo_python,bf16,,,SKIPPED (olmo_core not importable; pip install -e ./olmo-python)"
fi

# ---- zwt ----
# Run zwt_pretrain on the conf, parse every "step N ... <tps> tok/s" line.
# Drop the first 2 logged samples (graph-capture warmup + early ramp), then
# average the remainder. Sampling the last single step is too noisy on
# graph-captured runs because the per-step tok/s reflects whatever's
# happening at that exact instant, not steady state.
TMP=$(mktemp)
trap 'rm -f $TMP' EXIT
if [[ "$WORLD" -le 1 ]]; then
  "$BIN_ZWT" "$CONFIG" 2>"$TMP" >/dev/null || true
else
  # launch_ddp.sh tee's rank-0 stderr to logs/zwt_ddp_r0.log; we read that
  # for the steady-state samples after the run finishes.
  bash "$REPO/zwt/scripts/launch_ddp.sh" "$CONFIG" "$WORLD" >/dev/null 2>&1 || true
  cp logs/zwt_ddp_r0.log "$TMP" 2>/dev/null || true
fi

ZWT_TPS=$(grep -E "^step " "$TMP" \
          | awk '{ for (i=1;i<=NF;i++) if ($i ~ /tok\/s/) { print $(i-1); break } }' \
          | awk 'NR>2 { s+=$1; n++ } END { if (n) printf "%.0f", s/n; else print "" }')
ZWT_NOTE=$(grep -cE "^step " "$TMP")  # how many step lines we got
echo "zwt,bf16,,$ZWT_TPS,steady-state mean over $((ZWT_NOTE-2)) samples (world=$WORLD; see $TMP)"
