#!/bin/bash
# ============================================================================
# scripts/benchmark_optimizations.sh
#
# Three-way ablation of the C++ olmo_train binary on the SAME seed/data:
#   1. STANDARD       — baseline transformer (separate Q, K, V projections;
#                       separate gate and up FFN projections; no µP init)
#   2. FUSED          — fused QKV projection + fused gate+up FFN (one GEMM
#                       per pair). Same architecture, faster ops.
#   3. FUSED + µP     — fused ops AND µP initialization (Maximal Update
#                       Parametrization), which lets you tune LR on a small
#                       model and transfer to a larger one without retuning.
#
# Each variant runs with identical seed + batch + seq + steps so loss curves
# are directly comparable. Throughput and final loss are extracted from the
# tee'd logs and printed in a comparison table.
#
# Usage:
#   ./scripts/benchmark_optimizations.sh                # cpu, 50 steps
#   ./scripts/benchmark_optimizations.sh mps            # mps, 50 steps
#   ./scripts/benchmark_optimizations.sh cuda 200       # cuda, 200 steps
#
# --- Reads ---
#   $1 (device, optional) — cpu | mps | cuda. Default: cpu.
#   $2 (steps,  optional) — integer.          Default: 50.
#   ./build/olmo_train    — the C++ training binary (built if missing).
#
# --- Writes / Side effects ---
#   /tmp/bench_standard.txt      (full stdout of standard run)
#   /tmp/bench_fused.txt         (full stdout of fused run)
#   /tmp/bench_fused_mup.txt     (full stdout of fused+µP run)
#   May invoke `cmake --build build` to compile if the binary is missing.
#
# --- Calls ---
#   cmake --build build      (only if olmo_train is missing)
#   ./build/olmo_train       (3 invocations with progressively more flags)
#
# --- Role in workflow ---
#   Quick optimization sanity-check. Run after a successful build to see
#   that fused ops actually speed up training on your hardware AND don't
#   regress loss. Less heavyweight than scripts/benchmark.py (which also
#   compares to Python OLMo-core / cpp-llm).
# ============================================================================

set -euo pipefail
# Always run from the repo root regardless of where the user invoked us.
cd "$(dirname "$0")/.."

# Positional args with defaults: $1 = device, $2 = steps. The other knobs
# are intentionally fixed so runs are comparable across machines.
DEVICE="${1:-cpu}"
STEPS="${2:-50}"
SEED=42         # identical seed across the 3 runs ⇒ same RNG draws
BATCH_SIZE=4
SEQ_LEN=256

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  OLMo C++ Optimization Benchmark                           ║"
echo "║  Device: $DEVICE | Steps: $STEPS | Seed: $SEED             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check build exists; auto-build if not. We use sysctl on macOS, nproc on
# Linux, hence the `2>/dev/null || nproc` fallback.
if [ ! -f "build/olmo_train" ]; then
    echo "Building project..."
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1/3  STANDARD MODEL (baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Baseline: no --fused, no --mup. --profile turns on the per-op timer so
# the output has a "Throughput:" line that the summary block below greps.
./build/olmo_train --train --seed $SEED --profile \
    --steps $STEPS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN \
    --device $DEVICE --optimizer muon --lr 3e-4 --warmup-steps 10 \
    2>&1 | tee /tmp/bench_standard.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  2/3  FUSED MODEL (fused QKV + fused gate_up)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# +--fused: switches Transformer→FusedTransformer at construction time.
# Same parameter count as baseline; one-third the GEMM launches in attn.
./build/olmo_train --train --fused --seed $SEED --profile \
    --steps $STEPS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN \
    --device $DEVICE --optimizer muon --lr 3e-4 --warmup-steps 10 \
    2>&1 | tee /tmp/bench_fused.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  3/3  FUSED + µP MODEL (fused ops + µP initialization)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# +--mup: applies µ-Parametrization to weight init scales & LR multipliers.
# At this small size µP barely moves throughput, but it should leave loss
# very close to the baseline curve — the failure mode to look for is loss
# diverging or plateauing higher than the standard run.
./build/olmo_train --train --fused --mup --seed $SEED --profile \
    --steps $STEPS --batch-size $BATCH_SIZE --seq-len $SEQ_LEN \
    --device $DEVICE --optimizer muon --lr 3e-4 --warmup-steps 10 \
    2>&1 | tee /tmp/bench_fused_mup.txt

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  COMPARISON SUMMARY                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract throughput. olmo_train prints `Throughput: <N> tok/s` once at end;
# awk picks the second whitespace-separated field (the number).
STANDARD_TOKS=$(grep "Throughput:" /tmp/bench_standard.txt | awk '{print $2}')
FUSED_TOKS=$(grep "Throughput:" /tmp/bench_fused.txt | awk '{print $2}')
FUSED_MUP_TOKS=$(grep "Throughput:" /tmp/bench_fused_mup.txt | awk '{print $2}')

# Extract final loss from the last `Step …  loss: <x>` line. We tail -1 to
# get the last step, then grep -o just the `loss: 1.234` substring, then
# awk for the number. The combo is robust to trailing fields like tok/s.
STANDARD_LOSS=$(grep "Step" /tmp/bench_standard.txt | tail -1 | grep -o 'loss: [0-9.]*' | awk '{print $2}')
FUSED_LOSS=$(grep "Step" /tmp/bench_fused.txt | tail -1 | grep -o 'loss: [0-9.]*' | awk '{print $2}')
FUSED_MUP_LOSS=$(grep "Step" /tmp/bench_fused_mup.txt | tail -1 | grep -o 'loss: [0-9.]*' | awk '{print $2}')

printf "  %-25s %10s %10s\n" "Configuration" "tok/s" "final_loss"
printf "  %-25s %10s %10s\n" "─────────────────────────" "──────────" "──────────"
printf "  %-25s %10s %10s\n" "Standard" "${STANDARD_TOKS:-N/A}" "${STANDARD_LOSS:-N/A}"
printf "  %-25s %10s %10s\n" "Fused" "${FUSED_TOKS:-N/A}" "${FUSED_LOSS:-N/A}"
printf "  %-25s %10s %10s\n" "Fused + µP" "${FUSED_MUP_TOKS:-N/A}" "${FUSED_MUP_LOSS:-N/A}"

echo ""
echo "  Note: On CUDA with larger models (1B+), expect 5-20x speedup from"
echo "  fused ops + BF16 + Flash Attention (SDPA dispatch) + CUDA Graphs."
echo "  CPU/MPS improvements are more modest (1.2-2x) due to less kernel overhead."
echo ""
