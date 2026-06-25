#!/usr/bin/env bash
# prep_openwebtext.sh — prepare the OpenWebText dataset for Zero-Wait Trainer.
#
# Runs in three stages:
#   1. download the raw dataset from HuggingFace (cached locally)
#   2. tokenize with GPT-2 BPE into a single .npy file (u16 ids)
#   3. sanity-check by reading back a prefix and printing a histogram
#
# Disk requirements: ~40 GB for raw text, ~12 GB for tokenized .npy.
# Time: tokenization is CPU-bound; plan on ~15 minutes on a 32-core box.
#
# Usage:
#   zwt/scripts/prep_openwebtext.sh [out_dir]
#
# Defaults to ./data/owt.

set -euo pipefail

OUT_DIR="${1:-data/owt}"
RAW_DIR="${OUT_DIR}/raw"
TOKENS_OUT="${OUT_DIR}/owt_tokens.npy"
VOCAB_DIR="${OUT_DIR}/gpt2"

mkdir -p "${RAW_DIR}" "${VOCAB_DIR}"

# ------- 1. download the GPT-2 BPE vocab + merges -------
if [[ ! -f "${VOCAB_DIR}/vocab.json" ]]; then
  echo "[1/3] downloading GPT-2 BPE vocab..."
  curl -sL https://huggingface.co/gpt2/resolve/main/vocab.json \
       -o "${VOCAB_DIR}/vocab.json"
  curl -sL https://huggingface.co/gpt2/resolve/main/merges.txt \
       -o "${VOCAB_DIR}/merges.txt"
fi

# ------- 2. prepare_data handles the HF download + BPE in one shot -------
# The tool already supports --download-hf, so we just hand it Skylion007/openwebtext.
if [[ ! -f "${TOKENS_OUT}" ]]; then
  echo "[2/3] tokenizing OpenWebText -> ${TOKENS_OUT}"
  echo "      (this takes ~15 min on a 32-core box; longer elsewhere)"
  ./build/prepare_data \
    --download-hf Skylion007/openwebtext \
    --output "${TOKENS_OUT}" \
    --vocab-file "${VOCAB_DIR}/vocab.json" \
    --merges-file "${VOCAB_DIR}/merges.txt" \
    --threads "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
else
  echo "[2/3] tokens already exist at ${TOKENS_OUT}, skipping"
fi

# ------- 3. sanity check -------
echo "[3/3] sanity check..."
SIZE_BYTES=$(stat -f%z "${TOKENS_OUT}" 2>/dev/null || stat -c%s "${TOKENS_OUT}")
echo "      file size: $(( SIZE_BYTES / 1024 / 1024 )) MB"
echo "      token count (u16): $(( (SIZE_BYTES - 128) / 2 ))"

echo
echo "done. Point data.path in zwt/conf/owt_3B.conf at:"
echo "  ${TOKENS_OUT}"
