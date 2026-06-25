#!/usr/bin/env bash
# scripts/race/run_all.sh
#
# Orchestrator. Runs the full race pipeline end-to-end. Each phase
# bails on first failure so a broken correctness gate never bleeds
# into bogus throughput numbers.
#
#   00 env check
#   01 build C++
#   02 verify correctness (CPU + CUDA parity)
#   03 prepare tokenized data
#   04 train C++ side
#   05 train Python side
#   06 inference race — C++
#   07 inference race — Python
#   08 analyze + emit RESULT.md
#
# Skip individual phases by setting their step numbers in SKIP_PHASES:
#   SKIP_PHASES="05 07" bash scripts/race/run_all.sh   # skip Python sides
#   SKIP_PHASES="00 01" bash scripts/race/run_all.sh   # skip env+build (already done)
#
# Override build dir:
#   BUILD_DIR=build_h100 bash scripts/race/run_all.sh
#
# Each phase logs to scripts/race/results/<phase>/.

set -euo pipefail
cd "$(dirname "$0")/../.."
HERE="$(pwd)/scripts/race"

export BUILD_DIR="${BUILD_DIR:-build}"

skip_phases="${SKIP_PHASES:-}"
should_run() {
  for p in $skip_phases; do
    [[ "$p" == "$1" ]] && return 1
  done
  return 0
}

banner() {
  printf "\n\033[1;35m════════════════════════════════════════\033[0m\n"
  printf "\033[1;35m  PHASE %s — %s\033[0m\n" "$1" "$2"
  printf "\033[1;35m════════════════════════════════════════\033[0m\n"
}

phase() {
  local num="$1"; local name="$2"; local script="$3"; local soft="${4:-}"
  if should_run "$num"; then
    banner "$num" "$name"
    if [[ "$soft" == "soft" ]]; then
      # Comparison/baseline phases (Python OLMo, Python infer) depend on an
      # external env we don't control. Don't let them abort the whole run —
      # the C++ results and the final report must still be produced.
      if ! bash "$HERE/$script"; then
        printf "\n\033[1;33mWARNING: phase %s (%s) FAILED — continuing without it.\033[0m\n" "$num" "$name"
        printf "  (the C++ side and RESULT.md still complete; the comparison column will be blank)\n"
      fi
    else
      bash "$HERE/$script"
    fi
  else
    printf "\n\033[1;33mSKIPPED phase %s (%s)\033[0m\n" "$num" "$name"
  fi
}

mkdir -p "$HERE/results"
log="$HERE/results/run_all.log"
echo "[$(date -u +%FT%TZ)] run_all start" | tee -a "$log"

phase 00 "env check"           00_env_check.sh           2>&1 | tee -a "$log"
phase 01 "build C++"           01_build_cpp.sh           2>&1 | tee -a "$log"
phase 02 "verify correctness"  02_verify_correctness.sh  2>&1 | tee -a "$log"
phase 03 "prepare data"        03_prepare_data.sh        2>&1 | tee -a "$log"
phase 04 "train C++"           04_train_cpp.sh           2>&1 | tee -a "$log"
phase 05 "train Python"        05_train_python.sh   soft 2>&1 | tee -a "$log"
phase 06 "infer C++"           06_infer_cpp.sh           2>&1 | tee -a "$log"
phase 07 "infer Python"        07_infer_python.sh   soft 2>&1 | tee -a "$log"

banner 08 "analyze"
python3 "$HERE/08_analyze.py" 2>&1 | tee -a "$log"

printf "\n\033[1;32m✓  RACE COMPLETE\033[0m\n"
printf "Read:  %s/results/RESULT.md\n" "$HERE"
