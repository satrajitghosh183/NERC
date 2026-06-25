#!/usr/bin/env bash
# scripts/train_big.sh — ONE command to train a big model on all GPUs.
#
# Picks the right multi-GPU strategy automatically:
#   - Models that fit one GPU  -> DDP   (replicate + allreduce; validated)
#   - Models that don't         -> FSDP (shard optimizer/params across GPUs)
# and sets every required flag (gpu_data=0 streaming, cuda_graph=0, data/save
# paths, fsdp) so you don't hand-edit a conf. Just run it.
#
# Usage:
#   ./scripts/train_big.sh 3.6b              # 3.6B Muon via DDP  (the safe hero run)
#   ./scripts/train_big.sh 3b   500          # 2.75B AdamW via FSDP, 500 steps
#   ./scripts/train_big.sh 1b                # 1B AdamW via DDP   (quick sanity)
#   DATA=data/my.npy ./scripts/train_big.sh 3.6b 1000
#
# Env overrides: DATA (token .npy), STEPS, NGPU, MASTER_PORT.

set -uo pipefail
cd "$(dirname "$0")/.."

SIZE="${1:?usage: train_big.sh <1b|3b|3.6b> [steps]}"
STEPS="${2:-${STEPS:-200}}"
DATA="${DATA:-data/mg_tokens.npy}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

case "${SIZE,,}" in
  1b)   BASE=conf/olmo_1B_h100.conf;        FSDP=0; LABEL="1B AdamW · DDP" ;;
  3b)   BASE=conf/olmo_3B_h100_adamw.conf;  FSDP=1; LABEL="2.75B AdamW · FSDP (sharded)" ;;
  3.6b) BASE=conf/olmo_3B_h100.conf;        FSDP=0; LABEL="3.6B Muon · DDP (fits 1 GPU)" ;;
  *) echo "ERROR: unknown size '$SIZE' (use 1b | 3b | 3.6b)"; exit 1 ;;
esac

[[ -f "$BASE" ]]    || { echo "ERROR: base conf missing: $BASE"; exit 1; }
[[ -f "$DATA" ]]    || { echo "ERROR: data not found: $DATA  (set DATA=/path/to/tokens.npy)"; exit 1; }
[[ "$NGPU" -ge 2 ]] || { echo "ERROR: need >=2 GPUs (found $NGPU). 1 GPU: ./build/olmo_train $BASE"; exit 1; }
[[ -x build/olmo_train ]] || { echo "ERROR: build/olmo_train missing — ./scripts/build.sh --cuda --nccl"; exit 1; }

RUN="runs/big_${SIZE,,}"
OUT="conf/_big_${SIZE,,}.conf"
mkdir -p "$RUN"

# Override only the keys we must (parser accepts 'key = value'); keep model dims
# from the base conf. Replace-if-present for the core keys, then force fsdp.
sed -E \
  -e "s#^[[:space:]]*data_path([[:space:]=].*)?\$#data_path = ${DATA}#" \
  -e "s#^[[:space:]]*save([[:space:]=].*)?\$#save = ${RUN}/model.pt#" \
  -e "s#^[[:space:]]*heartbeat_path([[:space:]=].*)?\$#heartbeat_path = ${RUN}/hb.txt#" \
  -e "s#^[[:space:]]*steps([[:space:]=].*)?\$#steps = ${STEPS}#" \
  -e "s#^[[:space:]]*gpu_data([[:space:]=].*)?\$#gpu_data = 0#" \
  -e "s#^[[:space:]]*cuda_graph([[:space:]=].*)?\$#cuda_graph = 0#" \
  -e "s#^[[:space:]]*fsdp([[:space:]=].*)?\$#fsdp = ${FSDP}#" \
  "$BASE" > "$OUT"
# Ensure fsdp exists under [optimization] (sed above only edits it if present).
grep -qE '^[[:space:]]*fsdp[[:space:]=]' "$OUT" || \
  sed -i "/^\[optimization\]/a fsdp = ${FSDP}" "$OUT"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TRAIN BIG: ${LABEL}"
echo "║  base=${BASE}  gpus=${NGPU}  steps=${STEPS}  data=${DATA}"
echo "║  conf=${OUT}  ->  ${RUN}/"
echo "╚══════════════════════════════════════════════════════════╝"

exec ./scripts/launch_multigpu.sh "$OUT" "$NGPU"
