#!/usr/bin/env bash
# scripts/race/05_train_python.sh
#
# Python race side. Installs the in-repo OLMo-core (olmo-python/) if not
# already importable, then trains the matched llama2_271M architecture
# (n_layers=12, n_heads=8 → ~257M, same backbone as race_250m_cpp.conf)
# on data/race_tokens.npy via scripts/race/olmo_train_race.py.
#
# The trainer writes scripts/race/results/python_train/metrics.csv
# directly (step, loss, tok_per_s) — no log-scraping regex.

set -euo pipefail
cd "$(dirname "$0")/../.."

results_dir=scripts/race/results/python_train
mkdir -p "$results_dir"

say()  { printf "\033[1;36m[py]\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m[py]\033[0m %s\n" "$*"; exit 1; }

LOG="$results_dir/train.log"
METRICS="$results_dir/metrics.csv"

# ── Ensure OLMo-core is importable ──
if ! python3 -c "import olmo_core" 2>/dev/null; then
  if [[ -d olmo-python && -f olmo-python/pyproject.toml ]]; then
    say "installing in-repo OLMo-core (olmo-python/) — editable"
    # Modern Debian/Ubuntu/Pop OS mark the system Python "externally managed"
    # (PEP 668) and block `pip install` without a venv. Try the clean install
    # first; if PEP 668 blocks it, retry with --break-system-packages (this is
    # the in-repo OLMo-core, installed editable, so it only adds a .pth entry).
    if ! pip3 install -e olmo-python; then
      say "pip blocked (likely PEP 668 externally-managed env) — retrying with --break-system-packages"
      pip3 install --break-system-packages -e olmo-python || \
        fail "pip install -e olmo-python failed even with --break-system-packages.
       Best fix is a venv:
         python3 -m venv .race_venv && source .race_venv/bin/activate
         pip install -e olmo-python
       then re-run the race inside that venv."
    fi
  else
    fail "olmo-python/ not found and olmo_core not importable.
       Expected the Python OLMo-core at $(pwd)/olmo-python"
  fi
fi
say "olmo_core: $(python3 -c 'import olmo_core; print(olmo_core.__version__)' 2>/dev/null || echo '?')"

# ── Data must exist (phase 03) ──
[[ -f data/race_tokens.npy ]] || fail "data/race_tokens.npy missing — run 03_prepare_data.sh first"

# ── Train ──
# Single GPU: torchrun --nproc-per-node=1. OLMo-core needs the distributed
# env even on one rank (FSDP world_size=1).
say "training (torchrun, 1 rank); log → $LOG"
start=$(date +%s)
{
  echo "[run] $(date -u +%FT%TZ)"
  echo "[host] $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  torchrun --nproc-per-node=1 --standalone scripts/race/olmo_train_race.py \
      --data data/race_tokens.npy \
      --metrics-csv "$METRICS" \
      --save-folder "$results_dir/ckpt" \
      --steps 1000 \
      --warmup-steps 100 \
      --seq-len 1024 \
      --n-layers 12 \
      --n-heads 8 \
      --microbatch-instances 4 \
      --global-batch-instances 32 \
      --lr 3.0e-4 \
      --weight-decay 0.1 \
      --max-grad-norm 1.0 \
      --log-interval 10
} 2>&1 | tee "$LOG"
end=$(date +%s)
say "wall clock: $((end - start)) s"

if [[ -f "$METRICS" ]]; then
  python3 - <<EOF
import csv
rows = list(csv.DictReader(open("$METRICS")))
if rows:
    last = rows[-1]
    print(f"  {len(rows)} step records; final: step={last['step']} "
          f"loss={float(last['loss']):.4f} tok/s={last['tok_per_s']}")
else:
    print("  WARNING: metrics.csv is empty — check $LOG")
EOF
else
  say "WARNING: no metrics.csv produced — check $LOG"
fi
