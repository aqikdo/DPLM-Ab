#!/usr/bin/env bash
# LoRA finetune DPLM-2 650M on SAbDab nanobody tokens (8-GPU example).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"

python train.py \
  experiment=dplm2/dplm2_650m_sabdab_nanobody \
  name=dplm2_650m_sabdab_nanobody \
  datamodule.max_tokens=8192 \
  trainer.accumulate_grad_batches=1 \
  "$@"
