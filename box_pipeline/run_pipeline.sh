#!/usr/bin/env bash
# Hands-off shader-LM comparison pipeline. Runs sequentially (each stage needs both
# H100s). Watch: tail -f ~/pipeline/run.log  (or any ~/pipeline/0*.log).
set -uo pipefail
PIPE=~/pipeline; mkdir -p "$PIPE"
ML="$PIPE/run.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ML"; }

# clean slate
pkill -9 -f finetune_dora 2>/dev/null; pkill -9 -f olmo_train 2>/dev/null; sleep 2
say "=== PIPELINE START ==="
cd ~/OLMo-shader

# --- Stage 0: build the chat JSON config for the 3B SLM ---
python3 ~/shader_pipeline/conf2fulljson.py conf/shader_3B.conf configs/shader_3B.json 2>&1 | tee -a "$ML"

# --- Stage 1: train the from-scratch 3B SLM (long; resume=1 so it survives restarts) ---
say "STAGE 1: train 3B SLM  -> ~/pipeline/01_slm_train.log"
./scripts/launch_multigpu.sh conf/shader_3B.conf 2 > "$PIPE/01_slm_train.log" 2>&1
say "STAGE 1 done (rc=$?). $(grep -i loss "$PIPE/01_slm_train.log" | tail -1)"

# --- Stage 2: eval the SLM via llm-cpp chat (fast inference) ---
SLM=runs/shader_3B/model.pt; [ -f "$SLM" ] || SLM=runs/shader_3B/ckpt/latest.pt
say "STAGE 2: eval SLM ($SLM) -> ~/pipeline/02_slm_eval.log"
python3 ~/shader_pipeline/eval_slm.py "$SLM" configs/shader_3B.json slm > "$PIPE/02_slm_eval.log" 2>&1
say "STAGE 2 done (rc=$?). $(grep compile@1 "$PIPE/02_slm_eval.log" | tail -1)"

# --- Stage 3: DoRA-32B (Qwen2.5-Coder-32B, BF16 across both GPUs) ---
say "STAGE 3: DoRA-32B -> ~/pipeline/03_dora32b.log"
python3 ~/shader_pipeline/finetune_dora.py ~/models/qwen2.5-coder-32b 800 dora32b > "$PIPE/03_dora32b.log" 2>&1
say "STAGE 3 done (rc=$?). $(grep compile@1 "$PIPE/03_dora32b.log" | tail -1)"

# --- Stage 4: DoRA-3B (Qwen2.5-Coder-3B, same DoRA on our SLM's scale) ---
say "STAGE 4: download Qwen2.5-Coder-3B + DoRA-3B -> ~/pipeline/04_dora3b.log"
~/.local/bin/hf download Qwen/Qwen2.5-Coder-3B --local-dir ~/models/qwen2.5-coder-3b > "$PIPE/04_dora3b.log" 2>&1
python3 ~/shader_pipeline/finetune_dora.py ~/models/qwen2.5-coder-3b 800 dora3b >> "$PIPE/04_dora3b.log" 2>&1
say "STAGE 4 done (rc=$?). $(grep compile@1 "$PIPE/04_dora3b.log" | tail -1)"

# --- Stage 5: comparison report ---
say "STAGE 5: comparison report -> ~/pipeline/RESULTS.txt"
python3 ~/shader_pipeline/make_results.py 2>&1 | tee -a "$ML"
say "=== PIPELINE DONE ==="
