#!/usr/bin/env bash
# 50/50 fold+inv: 50% original noise + 50% progressive CDR (0.1→1.0 @1500).
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON="${PYTHON:-/home/zhaohongyan/miniconda3/envs/dplm/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
NAME="${1:-fold50inv50_hybrid_noise_epi01}"
EXP="dplm2_650m_vhh_mfdesign_antigen_fold_inv_hybrid_noise"

mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" -u train.py \
  experiment=dplm2/"${EXP}" \
  name="${NAME}" \
  datamodule.num_workers=0 \
  trainer.devices=1 \
  trainer.strategy=auto \
  "${@:2}" \
  > "logs/train_${NAME}.log" 2>&1 &

echo "Started PID=$! GPU=${GPU} name=${NAME}"
echo "Log: logs/train_${NAME}.log"
echo "Config: 50/50 fold+inv, 50% original + 50% CDR progressive @1500, max_steps=1500"
