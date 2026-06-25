#!/usr/bin/env bash
# scripts/race/02_verify_correctness.sh
#
# Validates the three correctness fixes on real CUDA hardware before
# we trust any throughput numbers from training:
#   1. WMMA store_matrix_sync UB fix
#   2. Half-rotation inverse RoPE fix
#   3. AutogradCUDA wrappers for olmo_ops kernels
# Plus the parity tests we run on CPU (which exercise different paths).
#
# Bails on any failure — do NOT train if these don't pass.

set -euo pipefail
cd "$(dirname "$0")/../.."

BUILD_DIR="${BUILD_DIR:-build}"
results_dir=scripts/race/results/correctness
mkdir -p "$results_dir"

say()  { printf "\033[1;36m[verify]\033[0m %s\n" "$*"; }
pass() { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

run_test() {
  local name="$1"; shift
  local log="$results_dir/${name}.log"
  if "$@" >"$log" 2>&1; then
    pass "$name"
  else
    tail -30 "$log"
    fail "$name (full log: $log)"
  fi
}

say "── CPU parity tests (sanity) ──"
run_test "test_fused_ce"        "$BUILD_DIR/test_fused_ce"
run_test "test_fused_qkv_rope"  "$BUILD_DIR/test_fused_qkv_rope"

say "── Infra tests ──"
run_test "test_paged_kv"        "$BUILD_DIR/test_paged_kv"
run_test "test_prefix_cache"    "$BUILD_DIR/test_prefix_cache"
run_test "test_scheduler"       "$BUILD_DIR/test_scheduler"

say "── CUDA parity (real GPU validation of the 3 correctness fixes) ──"
run_test "test_cuda_parity"     "$BUILD_DIR/test_cuda_parity"

printf "\n\033[1;32mALL CORRECTNESS GATES PASSED\033[0m\n"
printf "Logs: %s\n" "$results_dir"
