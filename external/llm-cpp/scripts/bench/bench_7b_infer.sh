#!/usr/bin/env bash
# scripts/bench/bench_7b_infer.sh
#
# End-to-end 7B inference benchmark: YOUR C++ engine vs ollama, SAME model
# (OLMo-2-1124-7B), same prompt shape, on this H100.
#
#   1. download  allenai/OLMo-2-1124-7B  (HF safetensors)        -> volume
#   2. convert   safetensors -> .pt  (convert_hf)                -> volume
#   3. quantize  .pt -> .int4.pt  (INT4 AWQ, matches ollama Q4)  -> volume   [optional]
#   4. bench C++ bench_chat at batch 1 / 8 / 32 (single + throughput)
#   5. bench ollama olmo2:7b (single-stream, --verbose eval rate)
#   6. print the comparison table
#
# The speed numbers do NOT need the OLMo tokenizer — bench_chat uses synthetic
# prompt tokens, so we pass GPT-2's vocab/merges only to satisfy the arg.
# (A coherent-text demo would need the real tokenizer; that's separate.)
#
# Everything heavy lives on the VOLUME (the 7B is ~15GB safetensors + ~14GB .pt
# + ~3.5GB int4; root disk has nowhere near that).
#
# Usage:
#   bash scripts/bench/bench_7b_infer.sh                 # full run
#   SKIP_INT4=1 bash scripts/bench/bench_7b_infer.sh     # bf16 only
#   HF_MODEL=allenai/OLMo-7B OLLAMA_TAG=olmo:7b bash scripts/bench/bench_7b_infer.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

# ── Config ──────────────────────────────────────────────────────────────────
BUILD_DIR="${BUILD_DIR:-build}"
HF_MODEL="${HF_MODEL:-allenai/OLMo-2-1124-7B}"
CONFIG="${CONFIG:-configs/olmo2_1124_7B.json}"
OLLAMA_TAG="${OLLAMA_TAG:-olmo2:7b}"
PROMPT_LEN="${PROMPT_LEN:-128}"
DECODE_LEN="${DECODE_LEN:-256}"

say()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
die()  { printf "\033[1;31mFAIL: %s\033[0m\n" "$*"; exit 1; }

[[ -x "$BUILD_DIR/convert_hf" ]] || die "$BUILD_DIR/convert_hf missing — ./scripts/build.sh --cuda"
[[ -x "$BUILD_DIR/bench_chat" ]] || die "$BUILD_DIR/bench_chat missing — ./scripts/build.sh --cuda"

# Storage: use the attached volume if present (H100 box), else a local dir
# (e.g. the 5060 Ti box, which has no /media/volume). Override with WORK=.
VOL="${VOL:-/media/volume/Prep_and_Voice_Training}"
if [[ -z "${WORK:-}" ]]; then
  if [[ -d "$VOL" ]]; then WORK="$VOL/bench7b"; else WORK="$HOME/bench7b"; fi
fi
HFDIR="$WORK/hf_model"; CKPT="$WORK/olmo2_7b.pt"; CKPT_INT4="$WORK/olmo2_7b.int4.pt"
RESULTS="$WORK/results"
mkdir -p "$WORK" "$RESULTS" || die "cannot create $WORK (set WORK=<dir with ~32GB free>)"
say "storage: $WORK ($(df -h "$WORK" | tail -1 | awk '{print $4}') free)"

# VRAM-aware: 7B bf16 (14GB) needs a big card. On <=24GB (e.g. 5060 Ti 16GB)
# skip bf16 and benchmark INT4 only (which fits and is the fair vs-ollama-Q4
# comparison anyway), with smaller batches.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "$VRAM_MB" && "$VRAM_MB" -lt 24000 ]]; then
  DO_BF16=0; BATCHES="${BATCHES:-1 4}"
  say "detected ${VRAM_MB}MB VRAM (<24GB) -> INT4 only, batches: $BATCHES (bf16 7B won't fit)"
else
  DO_BF16=1; BATCHES="${BATCHES:-1 8 32}"
fi

# GPT-2 tokenizer files — only needed to satisfy bench_chat's args (synthetic
# prompt tokens; tokenizer correctness does NOT affect the speed numbers).
VOCAB="data/gpt2/vocab.json"; MERGES="data/gpt2/merges.txt"
if [[ ! -f "$VOCAB" || ! -f "$MERGES" ]]; then
  mkdir -p data/gpt2
  curl -fsSL https://huggingface.co/gpt2/raw/main/vocab.json -o "$VOCAB" || die "vocab.json fetch failed"
  curl -fsSL https://huggingface.co/gpt2/raw/main/merges.txt -o "$MERGES" || die "merges.txt fetch failed"
fi

# ── 1. Download the HF model -> volume ──────────────────────────────────────
say "1. download $HF_MODEL -> $HFDIR"
if [[ ! -f "$HFDIR/config.json" ]]; then
  command -v huggingface-cli >/dev/null || pip3 install --break-system-packages -q huggingface_hub
  huggingface-cli download "$HF_MODEL" --local-dir "$HFDIR" \
    || die "HF download failed (login? 'huggingface-cli login')"
else
  say "   already downloaded"
fi

# ── 2. convert safetensors -> .pt ───────────────────────────────────────────
say "2. convert_hf -> $CKPT"
if [[ ! -f "$CKPT" ]]; then
  "$BUILD_DIR/convert_hf" --hf-dir "$HFDIR" --config "$CONFIG" --output "$CKPT" \
    || die "convert_hf failed — check $CONFIG matches the model's shapes (vocab/heads/ffn)"
fi
[[ -f "$CKPT" ]] || die "no converted checkpoint produced"

# ── 3. INT4 quantize (optional) ─────────────────────────────────────────────
if [[ -z "${SKIP_INT4:-}" && -x "$BUILD_DIR/quantize_int4" ]]; then
  say "3. quantize_int4 -> $CKPT_INT4"
  [[ -f "$CKPT_INT4" ]] || "$BUILD_DIR/quantize_int4" --in "$CKPT" --out "$CKPT_INT4" \
      --config "$CONFIG" --group-size 128 \
    || { echo "   (int4 quantize failed — continuing bf16-only; verify int4 load path)"; CKPT_INT4=""; }
else
  CKPT_INT4=""
fi

# ── 4. Benchmark YOUR engine (bench_chat) at several batch sizes ────────────
bench_cpp() {  # $1=label  $2=checkpoint  $3=batch
  local label="$1" ckpt="$2" b="$3" out="$RESULTS/cpp_${label}_b${b}.json"
  "$BUILD_DIR/bench_chat" --checkpoint "$ckpt" --config "$CONFIG" \
      --vocab-file "$VOCAB" --merges-file "$MERGES" --device cuda \
      --prompt-len "$PROMPT_LEN" --decode-len "$DECODE_LEN" --batch "$b" \
      --output "$out" 2>&1 | tee "$RESULTS/cpp_${label}_b${b}.log" \
    | grep -iE "Throughput|tok/s" || true
}

say "4. C++ bench_chat (decode_len=$DECODE_LEN)"
for b in $BATCHES; do
  if [[ "${DO_BF16:-1}" == "1" ]]; then echo "--- bf16, batch $b ---"; bench_cpp bf16 "$CKPT" "$b"; fi
  if [[ -n "$CKPT_INT4" ]]; then echo "--- int4, batch $b ---"; bench_cpp int4 "$CKPT_INT4" "$b"; fi
done

# ── 5. Benchmark ollama (single-stream) ─────────────────────────────────────
say "5. ollama $OLLAMA_TAG (single-stream)"
OLLAMA_TPS=""
if command -v ollama >/dev/null; then
  ollama pull "$OLLAMA_TAG" >/dev/null 2>&1 || echo "   (could not pull $OLLAMA_TAG — set OLLAMA_TAG)"
  # --verbose prints "eval rate: N tokens/s" to stderr
  OLLAMA_TPS=$(printf 'Write a long story about a robot.' \
      | ollama run "$OLLAMA_TAG" --verbose 2>&1 \
      | grep -iE "eval rate" | grep -oE "[0-9.]+ tokens/s" | head -1)
  echo "   ollama eval rate: ${OLLAMA_TPS:-<not parsed>}"
else
  echo "   ollama not installed — skipping (install from https://ollama.com)"
fi

# ── 6. Comparison table ─────────────────────────────────────────────────────
say "6. RESULT"
{
  echo "model:        $HF_MODEL  (C++)  vs  $OLLAMA_TAG (ollama)"
  echo "decode_len:   $DECODE_LEN   prompt_len: $PROMPT_LEN"
  echo "ollama single-stream: ${OLLAMA_TPS:-n/a}"
  echo
  printf "%-12s %-8s %-14s\n" "engine" "batch" "tok/s"
  for b in $BATCHES; do
    for label in bf16 int4; do
      f="$RESULTS/cpp_${label}_b${b}.log"
      [[ -f "$f" ]] || continue
      tps=$(grep -oiE "Throughput: *[0-9.]+" "$f" | grep -oE "[0-9.]+" | head -1)
      [[ -n "$tps" ]] && printf "cpp-%-8s %-8s %-14s\n" "$label" "$b" "$tps"
    done
  done
} | tee "$RESULTS/RESULT.txt"

say "done — full results in $RESULTS/  (RESULT.txt is the summary)"
echo "Reminder: report STEADY-STATE tok/s. The batched rows are your throughput edge;"
echo "the batch-1 row is the single-stream comparison vs ollama (expect parity at INT4)."
