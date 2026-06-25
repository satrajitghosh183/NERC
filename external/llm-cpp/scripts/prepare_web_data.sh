#!/usr/bin/env bash
# scripts/prepare_web_data.sh — build a real general-web-text corpus for training.
#
# Downloads N shards of C4 (allenai/c4, English, public), decompresses to JSONL,
# and tokenizes everything to a single GPT-2 .npy that conf/usable_1B.conf reads.
# Resumable: re-run to add more shards (already-fetched ones are skipped).
#
# Usage:
#   ./scripts/prepare_web_data.sh           # 40 shards (~6-8B tokens)
#   ./scripts/prepare_web_data.sh 80        # ~12-16B tokens (closer to Chinchilla-1B)
#   HF_TOKEN=hf_xxx ./scripts/prepare_web_data.sh 40   # if HF rate-limits you
#
# Tip: run it in the background and watch the log:
#   nohup ./scripts/prepare_web_data.sh 60 > prep.log 2>&1 &   ;  tail -f prep.log

set -uo pipefail
cd "$(dirname "$0")/.."

NSHARDS="${1:-40}"
OUT="${OUT:-data/web.npy}"
TOK="${TOK:-data/gpt2}"
RAW="${RAW:-data/c4_raw}"
BASE="https://huggingface.co/datasets/allenai/c4/resolve/main/en"
AUTH=(); [[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
mkdir -p "$TOK" "$RAW" "$(dirname "$OUT")"

[[ -x build/prepare_data ]] || { echo "ERROR: build/prepare_data missing — ./scripts/build.sh --cuda --nccl"; exit 1; }

# 1. GPT-2 tokenizer files (once).
[[ -s "$TOK/vocab.json" ]] || curl -fsSL "${AUTH[@]}" https://huggingface.co/gpt2/resolve/main/vocab.json -o "$TOK/vocab.json" || { echo "ERROR: vocab.json download failed"; exit 1; }
[[ -s "$TOK/merges.txt" ]] || curl -fsSL "${AUTH[@]}" https://huggingface.co/gpt2/resolve/main/merges.txt -o "$TOK/merges.txt" || { echo "ERROR: merges.txt download failed"; exit 1; }
echo "tokenizer: $TOK/{vocab.json,merges.txt} OK"

# 2. Download + decompress C4 shards.
ok=0
for ((i=0; i<NSHARDS; i++)); do
  gz=$(printf "c4-train.%05d-of-01024.json.gz" "$i")
  jf="$RAW/$(printf "c4-%05d.jsonl" "$i")"
  if [[ -s "$jf" ]]; then ok=$((ok+1)); continue; fi
  echo "[$((i+1))/$NSHARDS] $gz"
  if curl -fsSL "${AUTH[@]}" "$BASE/$gz" -o "$RAW/$gz" && gunzip -c "$RAW/$gz" > "$jf"; then
    rm -f "$RAW/$gz"; ok=$((ok+1))
  else
    echo "  WARN: shard $i failed (skipping) — if the FIRST one fails the URL/auth is wrong; set HF_TOKEN."
    rm -f "$RAW/$gz" "$jf"
  fi
done
[[ "$ok" -ge 1 ]] || { echo "ERROR: no shards downloaded — check network / HF_TOKEN"; exit 1; }
echo "downloaded/decompressed $ok shard(s) into $RAW"

# 3. Tokenize all JSONL shards -> one .npy (uses all cores).
echo "tokenizing -> $OUT (this can take 1-3h for billions of tokens)..."
./build/prepare_data --input "$RAW" --output "$OUT" \
  --vocab-file "$TOK/vocab.json" --merges-file "$TOK/merges.txt" \
  --max-tokens 100000000000 --threads "$(nproc)"

echo "DONE: $OUT  (you can 'rm -rf $RAW' to reclaim the JSONL disk space)"
echo "Now train:  ./scripts/launch_multigpu.sh conf/usable_1B.conf 2"
