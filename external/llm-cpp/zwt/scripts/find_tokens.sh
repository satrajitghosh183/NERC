#!/usr/bin/env bash
# find_tokens.sh — resolve a tokenized .npy path for zwt training.
#
# Resolution order:
#   1. $ZWT_TOKENS — direct file path. Used verbatim if it exists.
#   2. $ZWT_VOLUME — directory, searched first.
#   3. Default candidate roots (first hit wins):
#        /media/volume/Prep_and_Voice_Training   (Jetstream classic)
#        $HOME/data
#        /data
#        $(pwd)/downloads                          (where ./scripts/download_*.py lands)
#        $(pwd)/data                               (in-repo data dir)
#
# Search prefers .npy paths whose lower-cased name contains
# owt|openwebtext|tokens. Falls back to "largest .npy >= 1 GiB" if no
# name match. Largest file wins.
#
# On success: prints the resolved path on stdout, exit 0.
# On failure: helpful guidance on stderr, exit 1.

set -euo pipefail

# Direct path override.
if [[ -n "${ZWT_TOKENS:-}" ]]; then
  if [[ -f "$ZWT_TOKENS" ]]; then
    echo "$ZWT_TOKENS"
    exit 0
  fi
  echo "find_tokens: ZWT_TOKENS=$ZWT_TOKENS does not exist" >&2
  exit 1
fi

CANDIDATE_DIRS=()
[[ -n "${ZWT_VOLUME:-}" ]] && CANDIDATE_DIRS+=("$ZWT_VOLUME")
CANDIDATE_DIRS+=(
  "/media/volume/Prep_and_Voice_Training"
  "$HOME/data"
  "/data"
  "$(pwd)/downloads"
  "$(pwd)/data"
)

# Portable "stat size" — Linux is `-c %s`, macOS is `-f %z`.
_size() {
  if stat -c %s "$1" >/dev/null 2>&1; then
    stat -c '%s %n' "$1"
  else
    stat -f '%z %N' "$1"
  fi
}

found=""
for dir in "${CANDIDATE_DIRS[@]}"; do
  [[ -d "$dir" ]] || continue

  # Prefer name-matched npy (owt | openwebtext | tokens).
  match=$(
    find "$dir" -maxdepth 6 -type f -name '*.npy' 2>/dev/null \
      | awk 'tolower($0) ~ /(owt|openwebtext|tokens)/'
  )
  if [[ -z "$match" ]]; then
    # Fall back: any .npy >= 1 GiB. Tokenized OWT/Pile/etc. is always GiB+.
    match=$(find "$dir" -maxdepth 6 -type f -name '*.npy' -size +1G 2>/dev/null || true)
  fi

  # Pick largest by size.
  if [[ -n "$match" ]]; then
    biggest=""
    biggest_sz=0
    while IFS= read -r f; do
      [[ -f "$f" ]] || continue
      sz_line=$(_size "$f" 2>/dev/null || echo "")
      sz=${sz_line%% *}
      [[ -n "$sz" && "$sz" =~ ^[0-9]+$ ]] || continue
      if (( sz > biggest_sz )); then biggest_sz=$sz; biggest=$f; fi
    done <<< "$match"
    if [[ -n "$biggest" ]]; then
      found=$biggest
      break
    fi
  fi
done

if [[ -z "$found" ]]; then
  echo "find_tokens: no tokenized .npy found." >&2
  echo "  Searched dirs:" >&2
  for dir in "${CANDIDATE_DIRS[@]}"; do
    echo "    $dir $([[ -d $dir ]] && echo '(exists)' || echo '(missing)')" >&2
  done
  echo "" >&2
  echo "  Fix one of:" >&2
  echo "    export ZWT_TOKENS=/full/path/to/tokens.npy   # direct path" >&2
  echo "    export ZWT_VOLUME=/your/data/dir             # search root" >&2
  echo "    or place a *.npy under one of the default dirs above." >&2
  exit 1
fi

echo "$found"
