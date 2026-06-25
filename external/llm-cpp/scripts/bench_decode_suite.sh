#!/usr/bin/env bash
# scripts/bench_decode_suite.sh
#
# Reproducible decode benchmark suite (item 15.x). Runs the same prompt
# through `chat` under several configurations and emits a single
# CSV summarizing tokens/sec for each.
#
# Modes covered:
#   1. concat KV (legacy)               -- baseline
#   2. paged KV
#   3. paged KV + CUDA graph
#   4. paged KV + MTP speculative
#   5. paged KV + CUDA graph + MTP speculative
#   6. INT4-quantized weights + paged KV
#
# Usage:
#   ./scripts/bench_decode_suite.sh \
#     --checkpoint checkpoints/125M.pt \
#     --int4-ckpt checkpoints/125M.int4.pt \   # optional
#     --config configs/olmo2_125M.json \
#     --vocab data/gpt2/vocab.json \
#     --merges data/gpt2/merges.txt \
#     --max-tokens 256
#
# Output: results/decode_suite.csv

set -euo pipefail

CHECKPOINT=""
INT4_CKPT=""
CONFIG=""
VOCAB=""
MERGES=""
MAX_TOKENS=128
PROMPT="Once upon a time there was a little girl who"
OUT="results/decode_suite.csv"
DEVICE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --int4-ckpt)   INT4_CKPT="$2";  shift 2 ;;
    --config)      CONFIG="$2";     shift 2 ;;
    --vocab)       VOCAB="$2";      shift 2 ;;
    --merges)      MERGES="$2";     shift 2 ;;
    --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
    --prompt)      PROMPT="$2";     shift 2 ;;
    --out)         OUT="$2";        shift 2 ;;
    --device)      DEVICE="$2";     shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$OUT")"
echo "mode,tokens,tok_per_s" > "$OUT"

run_one() {
  local name="$1"; shift
  local out
  out=$(echo "$PROMPT" | ./build/chat --checkpoint "$CHECKPOINT" --config "$CONFIG" \
        --vocab-file "$VOCAB" --merges-file "$MERGES" \
        --temperature 0 --top-k 0 --top-p 1.0 --repetition-penalty 1.0 \
        --max-tokens "$MAX_TOKENS" --device "$DEVICE" "$@" 2>&1 \
        | grep -oE "[0-9]+ tokens, [0-9.]+ tok/s" | tail -1)
  local toks=$(echo "$out" | awk '{print $1}')
  local rate=$(echo "$out" | awk '{print $3}')
  echo "$name,$toks,$rate" >> "$OUT"
  echo "[$name] $out"
}

run_one "concat"                   --no-speculative
run_one "paged"                    --no-speculative --paged-kv
run_one "paged+graph"              --no-speculative --paged-kv --cuda-graph
run_one "paged+mtp_spec"           --paged-kv
run_one "paged+graph+mtp_spec"     --paged-kv --cuda-graph
if [[ -n "$INT4_CKPT" ]]; then
  run_one "int4+paged" --checkpoint "$INT4_CKPT" --no-speculative --paged-kv
fi

echo
echo "Suite complete. Results: $OUT"
cat "$OUT"
