#!/usr/bin/env bash
# 50/50 hybrid + epitope cross-attn hard mask. Twin of fold50inv50_hybrid_noise_epi01.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON="${PYTHON:-/home/zhaohongyan/miniconda3/envs/dplm/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NAME="${1:-fold50inv50_hybrid_epi_xattn}"
EXP="dplm2_650m_vhh_mfdesign_antigen_fold_inv_hybrid_epi_xattn"

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
echo "Config: 50/50 hybrid + epitope_cross_attn_mask=true, max_steps=1500"
