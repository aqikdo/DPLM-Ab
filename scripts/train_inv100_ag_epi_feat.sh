#!/usr/bin/env bash
# Pure Ag-conditioned invfold (inv=1.0) from no-Ag stage2; epi head + xattn feat.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYTHON="${PYTHON:-/home/zhaohongyan/miniconda3/envs/dplm/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NAME="${1:-inv100_ag_epi_feat}"
EXP="dplm2_650m_vhh_mfdesign_antigen_inv_only_epi_feat"
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
