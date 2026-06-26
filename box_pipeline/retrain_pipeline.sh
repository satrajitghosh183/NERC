#!/usr/bin/env bash
cd ~/OLMo-shader
echo "[retrain] waiting for tokenized npy + free GPUs..."
while [ ! -s ~/shader_data/corpus_v4/shaders_glslbpe.npy ]; do sleep 20; done
while pgrep -f synth_pipeline.sh >/dev/null; do sleep 60; done
echo "[retrain] START $(date)"
for sz in 300M 1B; do
  echo "[retrain] === shader_$sz ==="
  ./scripts/launch_multigpu.sh conf/shader_$sz.conf 2 > ~/pipeline/retrain_$sz.log 2>&1
  echo "[retrain] shader_$sz done rc=$? $(grep -a loss ~/pipeline/retrain_$sz.log | tail -1)"
done
echo "=== RETRAIN_DONE $(date) ==="
