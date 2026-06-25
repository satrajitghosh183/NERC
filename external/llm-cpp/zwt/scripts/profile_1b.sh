#!/usr/bin/env bash
# profile_1b.sh — nsys-profile a short 1B run to find where time is going.
#
# Runs zwt_pretrain with owt_1B_prof.conf (50 steps, no checkpoints) under
# Nsight Systems with CUDA + OS runtime tracing. Produces:
#
#   prof/zwt_1b_<timestamp>.nsys-rep     Open in Nsight Systems UI locally
#   prof/zwt_1b_<timestamp>.summary.txt  Terminal-readable kernel/API stats
#
# Usage:
#   bash zwt/scripts/profile_1b.sh              # full 50-step profile
#   bash zwt/scripts/profile_1b.sh --short      # skip first 15 sec (init)
#                                               # to keep trace smaller
#   bash zwt/scripts/profile_1b.sh --dmon       # skip nsys, watch nvidia-smi
#
# To open the .nsys-rep locally:
#   scp jetstream:.../olmo-python/prof/zwt_1b_*.nsys-rep .
#   nsight-sys zwt_1b_*.nsys-rep
# (install Nsight Systems: https://developer.nvidia.com/nsight-systems)

set -euo pipefail

CONF="zwt/conf/owt_1B_prof.conf"
BIN="./build/zwt_pretrain"
PROF_DIR="prof"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$PROF_DIR/zwt_1b_${TS}"

DO_SHORT=0
DO_DMON=0
for a in "$@"; do
  case "$a" in
    --short) DO_SHORT=1 ;;
    --dmon)  DO_DMON=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# Quick mode: just watch GPU utilization % while current run executes.
if [[ "$DO_DMON" -eq 1 ]]; then
  echo "Streaming GPU util — Ctrl-C to stop."
  echo "Columns: sm (SM busy %), mem (mem-controller busy %), enc/dec (ignore)."
  exec nvidia-smi dmon -s u -d 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — build first: bash ./scripts/build.sh --cuda" >&2
  exit 1
fi
if [[ ! -f "$CONF" ]]; then
  echo "missing $CONF" >&2
  exit 1
fi
if ! command -v nsys >/dev/null; then
  echo "nsys not found. On Jetstream CUDA 12 it ships with the toolkit:" >&2
  echo "  sudo apt-get install -y nsight-systems   # or just" >&2
  echo "  which nvcc && ls \$(dirname \$(which nvcc))/../nsight-systems-*/bin" >&2
  exit 1
fi

mkdir -p "$PROF_DIR"

# Make sure the data symlink exists (in case user hasn't run the launcher).
if [[ ! -e data/owt/owt_tokens.npy ]]; then
  echo "data/owt/owt_tokens.npy missing — run launch_1b_h100.sh --dry first" >&2
  exit 1
fi

# --- nsys args ---
# Using short-form flags to avoid ambiguous-prefix errors on older nsys
# (CUDA 12.0 ships nsys 2022.x where --sample prefix-matches a dozen opts).
#   -t : trace flavors (cuda = kernels+memcpy, osrt = OS runtime/blocking,
#        nvtx = harmless even if we don't emit any)
#   -s none : no CPU stack sampling (keeps trace smaller)
#   -o : output prefix (nsys appends .nsys-rep)
#   -y : delay seconds (skips init when --short)
NSYS_ARGS=(
  profile
  -t cuda,osrt,nvtx
  -s none
  -o "$OUT"
  --force-overwrite=true
)
if [[ "$DO_SHORT" -eq 1 ]]; then
  NSYS_ARGS+=( -y 15 )
fi

echo "=== nsys profile: $OUT.nsys-rep ==="
nsys "${NSYS_ARGS[@]}" -- "$BIN" "$CONF"

echo
echo "=== Stats summary: $OUT.summary.txt ==="
# Let nsys pick its default report set — names differ between versions.
# Default gives: gpukernsum, gpumemtimesum, gpumemsizesum, cudaapisum, osrtsum.
nsys stats \
  --format table "$OUT.nsys-rep" \
  2>&1 | tee "$OUT.summary.txt"

echo
echo "done."
echo "  report:   $OUT.nsys-rep"
echo "  summary:  $OUT.summary.txt"
echo
echo "Open the timeline locally:"
echo "  scp jetstream:$(pwd)/$OUT.nsys-rep ."
echo "  nsight-sys $(basename "$OUT").nsys-rep"
