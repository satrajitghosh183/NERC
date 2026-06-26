#!/usr/bin/env bash
cd ~/shader_data/synth
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$HOME/vllm_env/bin:$PATH
echo "[synth] generating..."; ~/vllm_env/bin/python ~/shader_pipeline/gen_shaders.py --model ~/models/qwen2.5-coder-32b --n-prompts 10000 --k 2 --out gen_full.jsonl
echo "[synth] labeling..."; python3 ~/shader_pipeline/label_shaders.py --in gen_full.jsonl --labeled labeled_full.jsonl --corpus-out synth_corpus_full.txt --workers 32
echo "[synth] repairing..."; ~/vllm_env/bin/python ~/shader_pipeline/repair_loop.py --in labeled_full.jsonl --model ~/models/qwen2.5-coder-32b --out repaired_full.jsonl --workers 32
echo "=== SYNTH_DONE ==="
