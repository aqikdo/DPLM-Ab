#!/usr/bin/env bash
set -euo pipefail
cd /AIRvePFS/dair/zhaohongyan/dplm
LOG=logs/train_fold50inv50_cdr_sched_wait.log
echo "$(date) waiter start" >> "$LOG"
while true; do
  if pgrep -f 'name=fold50inv50_cdr_sched_epi01' >/dev/null 2>&1; then
    echo "$(date) already training" >> "$LOG"
    exit 0
  fi
  if [[ -f logs/fold50inv50_cdr_sched_epi01/checkpoints/every250/step_1500.ckpt ]]; then
    echo "$(date) already has step_1500" >> "$LOG"
    exit 0
  fi
  if ! pgrep -f 'eval_antigen_cond_nano43.py' >/dev/null 2>&1; then
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    echo "$(date) no eval; GPU1 used=${used}MiB" >> "$LOG"
    if [[ "${used:-99999}" -lt 3000 ]]; then
      echo "$(date) launching fold50inv50_cdr_sched on GPU1" >> "$LOG"
      CUDA_VISIBLE_DEVICES=1 bash scripts/train_fold50inv50_cdr_schedule.sh fold50inv50_cdr_sched_epi01 >> "$LOG" 2>&1
      exit 0
    fi
  else
    echo "$(date) still waiting (eval running)" >> "$LOG"
  fi
  sleep 120
done
