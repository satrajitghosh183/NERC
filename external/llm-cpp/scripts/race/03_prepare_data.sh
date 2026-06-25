#!/usr/bin/env bash
# scripts/race/03_prepare_data.sh
#
# Tokenizes a fixed corpus to data/race_tokens.npy. Both C++ and Python
# sides read this same file, so neither has a tokenization-speed advantage
# (or different vocabulary).
#
# Default corpus: TinyStories (small, well-known, fast to download). The
# C++ prepare_data tool produces a contiguous uint16 .npy that the Python
# trainer can memmap directly via numpy.load.

set -euo pipefail
cd "$(dirname "$0")/../.."

say() { printf "\033[1;36m[data]\033[0m %s\n" "$*"; }

BUILD_DIR="${BUILD_DIR:-build}"
OUT=data/race_tokens.npy
VOCAB=data/gpt2/vocab.json
MERGES=data/gpt2/merges.txt

mkdir -p data/gpt2

# Fetch the GPT-2 tokenizer once. Both sides use GPT-2 BPE.
if [[ ! -f "$VOCAB" || ! -f "$MERGES" ]]; then
  say "downloading GPT-2 tokenizer"
  # -f makes curl FAIL on HTTP errors instead of silently writing the error
  # page into the file (which prepare_data would then tokenize as garbage).
  curl -fsSL https://huggingface.co/gpt2/raw/main/vocab.json -o "$VOCAB" \
    || { say "ERROR: failed to download vocab.json"; rm -f "$VOCAB"; exit 1; }
  curl -fsSL https://huggingface.co/gpt2/raw/main/merges.txt -o "$MERGES" \
    || { say "ERROR: failed to download merges.txt"; rm -f "$MERGES"; exit 1; }
  python3 -c "import json; json.load(open('$VOCAB'))" \
    || { say "ERROR: vocab.json not valid JSON (corrupt download)"; rm -f "$VOCAB"; exit 1; }
  if [[ "$(wc -l < "$MERGES")" -lt 100 ]]; then
    say "ERROR: merges.txt too small — corrupt download"; rm -f "$MERGES"; exit 1
  fi
  say "tokenizer OK"
fi

if [[ -f "$OUT" ]]; then
  size=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
  say "data/race_tokens.npy exists ($((size/1024/1024)) MB) — skipping tokenization"
else
  say "tokenizing TinyStories → $OUT (~5 min first time)"
  "$BUILD_DIR/prepare_data" \
    --download-hf roneneldan/TinyStories \
    --output "$OUT" \
    --vocab-file "$VOCAB" \
    --merges-file "$MERGES"
fi

# Report tokens stat.
python3 - <<EOF
import numpy as np, os
arr = np.load("$OUT", mmap_mode="r")
print(f"  tokens: {arr.shape[0]:,}  dtype: {arr.dtype}  size: {os.path.getsize('$OUT')/1024/1024:.1f} MB")
EOF
