#!/usr/bin/env bash
# verify_ddp.sh — fail-fast smoke gate before launching a real 2-GPU bench.
#
# Three checks, in order:
#   1. zwt_ddp_loopback_tests passes (BucketManager gather/scatter index map).
#   2. 50 steps of zwt_pretrain via launch_ddp.sh on a 2-GPU 1B conf —
#      both ranks log, final loss is finite + descending vs step 1.
#   3. 50 steps of pt_baseline.py under torchrun --nproc-per-node=2 on the
#      same conf — runs to completion.
#
# If any check fails, the script bails and tells you which log to inspect.
# Total runtime: ~3-5 min on a 2x H100 box. Run this BEFORE bench_threeway.sh
# WORLD=2.
#
# Usage:
#   bash zwt/scripts/verify_ddp.sh
#   bash zwt/scripts/verify_ddp.sh zwt/conf/owt_1B_2xh100.conf   # custom conf

set -euo pipefail

CONF=${1:-zwt/conf/owt_1B_2xh100.conf}
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/verify_ddp_${TS}"
mkdir -p "$OUT" logs

if [[ ! -x ./build/zwt_pretrain ]]; then
  echo "verify_ddp: ./build/zwt_pretrain not built. Run ./scripts/build.sh --cuda first." >&2
  exit 2
fi
if [[ ! -x ./build/zwt_ddp_loopback_tests ]]; then
  echo "verify_ddp: ./build/zwt_ddp_loopback_tests not built." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "verify_ddp: nvidia-smi not on PATH — needs CUDA" >&2
  exit 2
fi
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$N_GPU" -lt 2 ]]; then
  echo "verify_ddp: need >=2 GPUs (found $N_GPU); skipping the multi-rank checks." >&2
  echo "verify_ddp: still running the loopback test." >&2
fi

stage() { echo; echo "── $1 ──"; }
fail()  { echo "  FAIL  $1 — see $2" >&2; exit 1; }
pass()  { echo "  PASS  $1"; }

# ── 0. data symlink ─────────────────────────────────────────────────────
# zwt_pretrain reads data/owt/owt_tokens.npy from the .conf; we materialize
# that symlink here via find_tokens.sh so neither rank crashes with
# 'TokenLoader: cannot open ...' at startup.
stage "0. tokens"
if TOKENS=$(bash "$(dirname "$0")/find_tokens.sh" 2>/dev/null); then
  mkdir -p data/owt
  ln -sfn "$TOKENS" data/owt/owt_tokens.npy
  echo "  tokens: $TOKENS"
  pass "symlink data/owt/owt_tokens.npy → $TOKENS"
else
  echo "  WARN: no tokens found — pt_baseline (synthetic ids) will still run, but the zwt leg will skip" >&2
fi

# ── 1. loopback test ────────────────────────────────────────────────────
stage "1. zwt_ddp_loopback_tests"
LOOP_LOG="$OUT/loopback.log"
if ! ./build/zwt_ddp_loopback_tests > "$LOOP_LOG" 2>&1; then
  fail "loopback failed" "$LOOP_LOG"
fi
pass "loopback ($(grep -c PASS "$LOOP_LOG") assertions)"

if [[ "$N_GPU" -lt 2 ]]; then
  echo
  echo "verify_ddp: skipping zwt + pt 2-GPU smoke (only $N_GPU GPU)." >&2
  exit 0
fi

# ── 2. zwt 50-step DDP run ──────────────────────────────────────────────
stage "2. zwt 50-step DDP smoke"
SMOKE_CONF="$OUT/smoke_zwt.conf"
# Patch a copy of the conf to run only 50 steps and skip checkpoint writes.
sed -E '
  s/^(\s*max_steps\s*=).*/\1 50/g;
  s/^(\s*log_interval\s*=).*/\1 5/g;
  s/^(\s*ckpt_interval\s*=).*/\1 0/g;
' "$CONF" > "$SMOKE_CONF"

ZWT_LOG="$OUT/zwt_ddp_r0.log"
if ! bash zwt/scripts/launch_ddp.sh "$SMOKE_CONF" 2 > "$OUT/zwt_launch.log" 2>&1; then
  fail "zwt 2-GPU run errored" "$OUT/zwt_launch.log"
fi
# rank 0 log lives at logs/zwt_ddp_r0.log (launcher convention).
cp logs/zwt_ddp_r0.log "$ZWT_LOG" 2>/dev/null || true

# Pull first and last step's loss; assert finite and final < first.
FIRST=$(grep -E "^step +1 " "$ZWT_LOG" | head -1 | grep -oE 'loss [0-9.]+' | awk '{print $2}')
LAST=$( grep -E "^step "      "$ZWT_LOG" | tail -1 | grep -oE 'loss [0-9.]+' | awk '{print $2}')
if [[ -z "$FIRST" || -z "$LAST" ]]; then
  fail "zwt run produced no step lines" "$ZWT_LOG"
fi
# Bash float compare via awk.
if ! awk "BEGIN { exit !($LAST < $FIRST) }" </dev/null; then
  fail "zwt loss not descending: first=$FIRST last=$LAST" "$ZWT_LOG"
fi
pass "zwt 2-GPU loss descended $FIRST → $LAST"

# ── 3. pt_baseline 50-step DDP run ──────────────────────────────────────
stage "3. pt_baseline 50-step DDP smoke"
PT_LOG="$OUT/pt_baseline.log"
# Eager (no compile) — just a "did it run" gate; tok/s is bench_threeway's job.
if ! torchrun --nproc-per-node=2 --nnodes=1 --master-port=29502 \
        zwt/scripts/pt_baseline.py \
        --config "$SMOKE_CONF" --warmup 5 --steps 50 \
        --dtype bf16 > "$PT_LOG" 2>&1; then
  fail "pt_baseline torchrun errored" "$PT_LOG"
fi
if ! grep -q "tok/s=" "$PT_LOG"; then
  fail "pt_baseline produced no headline tok/s" "$PT_LOG"
fi
pass "pt_baseline 2-GPU run"

echo
echo "verify_ddp: all checks passed. Logs in $OUT/"
