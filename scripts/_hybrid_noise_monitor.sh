#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT=$PWD PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
LOG=logs/hybrid_noise_train_eval_monitor.log
CKPT_B=logs/fold_B_hybrid_noise_epi01/checkpoints/every250/step_1500.ckpt
CKPT_50=logs/fold50inv50_hybrid_noise_epi01/checkpoints/every250/step_1500.ckpt
OUT=outputs/vhh_eval_hybrid_noise_vs_baselines
echo "$(date) monitor start" | tee -a "$LOG"
while true; do
  b=0; f=0
  [[ -f "$CKPT_B" ]] && b=1
  [[ -f "$CKPT_50" ]] && f=1
  sb=$(grep -oE "global step [0-9]+" logs/train_fold_B_hybrid_noise_epi01.log 2>/dev/null | tail -1 || true)
  s50=$(grep -oE "global step [0-9]+" logs/train_fold50inv50_hybrid_noise_epi01.log 2>/dev/null | tail -1 || true)
  echo "$(date) B=$b($sb) 50=$f($s50)" | tee -a "$LOG"
  if [[ $b -eq 1 && $f -eq 1 ]]; then
    sleep 40
    mkdir -p "$OUT"
    [[ -d outputs/vhh_eval_fold_B1500_fr_modes/fold_B_step1500 && ! -e "$OUT/fold_B_step1500" ]] && cp -a outputs/vhh_eval_fold_B1500_fr_modes/fold_B_step1500 "$OUT/fold_B_step1500"
    [[ -d outputs/vhh_eval_fold_B1500_vs_fold50inv50/fold50inv50_step1500 && ! -e "$OUT/fold50inv50_step1500" ]] && cp -a outputs/vhh_eval_fold_B1500_vs_fold50inv50/fold50inv50_step1500 "$OUT/fold50inv50_step1500"
    echo "$(date) evaluating hybrids" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 /home/zhaohongyan/miniconda3/envs/dplm/bin/python -u scripts/eval_antigen_cond_nano43.py \
      --out-root "$OUT" --max-iter 100 \
      --modes folding,inverse_folding,fr_folding,fr_inverse_folding \
      --ckpts \
        fold_B_hybrid_noise_step1500=$CKPT_B \
        fold50inv50_hybrid_noise_step1500=$CKPT_50 \
      2>&1 | tee -a "$LOG"
    /home/zhaohongyan/miniconda3/envs/dplm/bin/python scripts/merge_hybrid_noise_comparison.py | tee -a "$LOG"
    echo "$(date) EVAL_DONE" | tee -a "$LOG"
    exit 0
  fi
  sleep 180
done
