#!/usr/bin/env bash
# scripts/disk_cleanup.sh — free root-disk space FAST and safely.
#
# Phase 1: report the biggest consumers.
# Phase 2: purge regenerable caches (always safe).
# Phase 3: list the big MOVABLE items (model downloads, checkpoints) with exact
#          commands — NOT auto-deleted, you confirm.
#
# Usage:  bash scripts/disk_cleanup.sh

set -uo pipefail
cd "$(dirname "$0")/.."
VOL="${VOL:-/media/volume/Prep_and_Voice_Training}"

hr() { printf '%.0s─' {1..70}; echo; }
say() { printf "\033[1;36m%s\033[0m\n" "$*"; }

say "Root disk before:"; df -h / | tail -1; hr

# ── Phase 1: biggest space users under $HOME (top 20) ──────────────────────
say "Top space users under $HOME (be patient)…"
du -x -h -d 3 "$HOME" 2>/dev/null | sort -rh | head -20
hr

# ── Phase 2: SAFE purges (all regenerable) ─────────────────────────────────
say "Purging regenerable caches…"
rm -rf "$HOME/.cache/pip" 2>/dev/null && echo "  cleared pip cache"
rm -rf /tmp/race-dataset-cache 2>/dev/null && echo "  cleared /tmp/race-dataset-cache"
rm -rf /tmp/torchinductor_* /tmp/tmp* 2>/dev/null || true
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null && echo "  cleared __pycache__"
# OLMo-core writes sharded step checkpoints under the race results — regenerable.
rm -rf scripts/race/results/python_train/ckpt 2>/dev/null && echo "  cleared python_train/ckpt"
command -v pip3 >/dev/null && pip3 cache purge >/dev/null 2>&1 || true
# System logs + apt cache (need sudo; ignore if not available).
sudo journalctl --vacuum-size=200M >/dev/null 2>&1 && echo "  vacuumed journal to 200M" || true
sudo apt-get clean >/dev/null 2>&1 && echo "  cleared apt cache" || true
hr

# ── Phase 3: BIG movable items — review and act manually ───────────────────
say "Big items you should MOVE to the volume or DELETE (not auto-removed):"
echo
for d in olmo-32b checkpoints data scripts/race/results; do
  if [[ -e "$d" ]]; then
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    printf "  %-26s %s\n" "$d" "$sz"
  fi
done
echo
echo "Suggested actions (run the ones that apply):"
echo "  # 60GB+ model download you don't need to train — delete or move to volume:"
echo "  rm -rf olmo-32b                              # if you don't need it"
echo "  #   or: mv olmo-32b $VOL/                    # keep it, on the volume"
echo
echo "  # Old root-disk checkpoints (future runs already write to the volume):"
echo "  mv checkpoints $VOL/old_checkpoints 2>/dev/null || true"
echo
echo "  # Training data sitting on root (TinyStories etc.) — move to the volume:"
echo "  mv data/*.npy $VOL/data/ 2>/dev/null || true"
hr
say "Root disk after Phase 2 purge:"; df -h / | tail -1
echo
say "If still tight, act on the Phase-3 items above (esp. olmo-32b)."
