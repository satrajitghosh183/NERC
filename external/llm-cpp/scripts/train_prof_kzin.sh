# ============================================================================
# scripts/train_prof_kzin.sh
#
# One-shot Nsight Systems trace of a 125M-parameter olmo_train run on the kzin
# Linux box. The output report (train_30.nsys-rep / .qdstrm / .sqlite) can be
# opened in nsys-ui or post-processed via `nsys stats` to inspect CUDA kernel
# timelines, NVTX ranges emitted by the trainer, OS runtime calls, etc.
#
# This file intentionally has no shebang and no `set -e` — it is a single
# command meant to be run interactively (`bash scripts/train_prof_kzin.sh`)
# or copy-pasted. Shell exit status is the nsys exit status.
#
# Usage:
#   bash scripts/train_prof_kzin.sh
#
# --- Reads ---
#   conf/olmo_prof_125M.conf  (training config: model size, steps, data path)
#
# --- Writes / Side effects ---
#   train_30.nsys-rep   (Nsight Systems profile, primary output)
#   train_30.sqlite     (companion sqlite DB; created when --stats=true)
#   stdout              (per-kernel summary tables from --stats=true)
#
# --- Calls ---
#   nsys                  (NVIDIA Nsight Systems CLI)
#   ./build/olmo_train    (the C++ training binary, must be CUDA-built)
#
# --- Role in workflow ---
#   Ad-hoc profiling tool. Run after a successful build to see where the
#   training step actually spends time on the GPU. Not part of the
#   automated benchmark suite — use scripts/jetstream_benchmark.sh for that.
#
# --- Flag notes ---
#   --stats=true            Emit per-kernel summary tables to stdout AND
#                           create the .sqlite DB (slows post-processing
#                           but is the easiest way to get numbers).
#   --force-overwrite true  Overwrite a previous train_30.* report instead
#                           of erroring out (handy for repeat runs).
#   --output=train_30       Base name for all generated report files.
# ============================================================================

# Profile a single olmo_train run with the 125M-parameter profiling config.
# nsys spawns olmo_train as a child, samples CUDA + OSRT, then summarizes.
nsys profile --stats=true --force-overwrite true --output=train_30 ./build/olmo_train conf/olmo_prof_125M.conf

