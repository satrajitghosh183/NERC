#!/usr/bin/env bash
# scripts/bench_subq.sh
#
# SubQ head-to-head bench harness. Runs `bench_attn` across the seq lengths
# and shapes that match SubQ's published headline ("52x over FlashAttention-2
# at 1M tokens, 7.2x at 128K, 13.2x at 256K, 23.0x at 512K") and writes the
# raw CSV that downstream analysis decomposes.
#
# This is the "before" picture — every later improvement (FA-3 enable,
# paged KV layout, content sparsity, fused selector+SDPA) gets compared
# back to results/subq_baseline.csv.
#
# Shape: 1B-class transformer (H=32, Hkv=8, D=128). Matches typical
# Llama-3-1B / SubQ-class architectures.
#
# Modes run:
#   dense — torch::scaled_dot_product_attention. On H100 + recent libtorch
#           this dispatches to FlashAttention-2 internally, so this row is
#           our "FA-2 baseline" for the SubQ comparison.
#   fa2   — explicit FA-2 path (decode only for now; prefill falls through
#           to dense until we ship a standalone FA-2 prefill kernel).
#   fa3   — currently stub_falls_to_dense. Will become a separate row once
#           we wire FA-3 (either via a vendored kernel or via libtorch's
#           SDPA backend selection on Hopper).
#
# Stages run:
#   prefill at [8K, 32K, 128K, 256K, 512K, 1M]
#   decode  at KV cache lens [16K, 64K, 256K] with 64 single-query steps
#
# Output: results/subq_baseline.csv (one row per (mode, stage, T)).
#
# Usage:
#   ./scripts/bench_subq.sh                         # H100 sweep
#   ./scripts/bench_subq.sh --short                 # quick subset (smoke)
#   ./scripts/bench_subq.sh --device cpu --short    # local macOS sanity

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO_ROOT/build/bench_attn"
RESULTS_DIR="$REPO_ROOT/results"
OUT="$RESULTS_DIR/subq_baseline.csv"
DEVICE="auto"
DTYPE="auto"
WARMUP=3
ITERS=5
SHORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --dtype)  DTYPE="$2";  shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --iters)  ITERS="$2";  shift 2;;
    --output) OUT="$2";    shift 2;;
    --short)  SHORT=1;     shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ ! -x "$BENCH" ]]; then
  echo "bench_attn not built. Run:" >&2
  echo "  cmake --build $REPO_ROOT/build --target bench_attn -j" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

# Sweep configuration. SubQ's reported numbers are at 1M tokens single
# sequence, B=1. Smaller seq lens give us the curve, not just the headline.
if [[ $SHORT -eq 1 ]]; then
  PREFILL_LENS="1024,4096"
  DECODE_LENS="1024,4096"
  DECODE_STEPS=4
else
  PREFILL_LENS="8192,32768,131072,262144,524288,1048576"
  DECODE_LENS="16384,65536,262144"
  DECODE_STEPS=64
fi

# 1B-class shapes.
B=1
HEADS=32
KVHEADS=8
HEAD_DIM=128

echo "=== bench_subq ==="
echo "  device=$DEVICE dtype=$DTYPE warmup=$WARMUP iters=$ITERS"
echo "  shape: B=$B H=$HEADS Hkv=$KVHEADS D=$HEAD_DIM"
echo "  prefill lens: $PREFILL_LENS"
echo "  decode  lens: $DECODE_LENS (steps=$DECODE_STEPS)"
echo "  output: $OUT"
echo

# Fresh CSV with one header.
: > "$OUT.tmp"

run_mode() {
  local mode="$1" stage="$2" lens="$3"
  echo "--- mode=$mode stage=$stage ---"
  local extra=""
  if [[ "$stage" == "decode" ]]; then
    extra="--decode-steps $DECODE_STEPS"
  fi
  # bench_attn writes its own CSV when --output is given; we capture the
  # row(s) and append to $OUT.tmp. The first invocation includes the header;
  # subsequent invocations' headers are stripped.
  local tmp; tmp="$(mktemp /tmp/bench_attn.XXXXXX.csv)"
  "$BENCH" \
    --mode "$mode" --stage "$stage" \
    --seq-lens "$lens" \
    --batch "$B" --n-heads "$HEADS" --n-kv-heads "$KVHEADS" --head-dim "$HEAD_DIM" \
    --device "$DEVICE" --dtype "$DTYPE" \
    --warmup "$WARMUP" --iters "$ITERS" \
    $extra \
    --output "$tmp"
  if [[ ! -s "$OUT.tmp" ]]; then
    cat "$tmp" >> "$OUT.tmp"
  else
    tail -n +2 "$tmp" >> "$OUT.tmp"
  fi
  rm -f "$tmp"
}

# Prefill: dense + fa3 (stub). fa2 prefill currently falls through to dense
# in the bench tool, so skip duplicating it until we ship a real FA-2
# prefill kernel — we'd just be running the same code path twice.
run_mode dense prefill "$PREFILL_LENS"
run_mode fa3   prefill "$PREFILL_LENS"

# Decode: dense + fa2 (real, via flash_decode) + fa3 (stub).
run_mode dense decode "$DECODE_LENS"
run_mode fa2   decode "$DECODE_LENS"
run_mode fa3   decode "$DECODE_LENS"

mv "$OUT.tmp" "$OUT"
echo
echo "=== done ==="
echo "Wrote $OUT"
echo
echo "Quick view:"
column -s, -t < "$OUT" | sed 's/^/  /'
