#!/usr/bin/env bash
# Rerun epi_feat twin to verify restored code matches prior nano43 metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYTHON="${PYTHON:-/home/zhaohongyan/miniconda3/envs/dplm/bin/python}"
NAME="${1:-fold50inv50_hybrid_epi_feat_rerun}"
CKPT="logs/${NAME}/checkpoints/every250/step_1500.ckpt"
OUT="outputs/vhh_eval_hybrid_epi_feat_rerun"
LOG="logs/eval_${NAME}.log"
GPU_EVAL="${CUDA_VISIBLE_DEVICES:-1}"
PREV="outputs/vhh_eval_hybrid_epi_feat/fold50inv50_hybrid_epi_feat_step1500/summary.json"

echo "[$(date)] waiting for ${CKPT}" | tee -a "$LOG"
while [[ ! -f "$CKPT" ]]; do sleep 60; done
sleep 30
echo "[$(date)] eval on GPU ${GPU_EVAL}" | tee -a "$LOG"

mkdir -p "$OUT"
env CUDA_VISIBLE_DEVICES="${GPU_EVAL}" PYTHONUNBUFFERED=1 "${PYTHON}" -u scripts/eval_antigen_cond_nano43.py \
  --out-root "$OUT" \
  --antigen-max-len 256 \
  --modes folding,inverse_folding,fr_folding,fr_inverse_folding \
  --ckpts "${NAME}_step1500=${CKPT}" \
  >> "$LOG" 2>&1

"${PYTHON}" - <<PY | tee -a "$LOG"
import json
from pathlib import Path

prev = json.loads(Path("${PREV}").read_text())
newp = Path("${OUT}") / "${NAME}_step1500" / "summary.json"
new = json.loads(newp.read_text())
new["tag"] = newp.parent.name

def fmt_fold(m):
    regs = m.get("regions_ca_rmsd") or {}
    return (
        f"CA={m.get('ca_rmsd'):.4f} TM={m.get('bb_tmscore'):.4f} "
        f"CDR3={regs.get('CDR3'):.4f}"
    )

def fmt_aar(m):
    regs = m.get("regions") or {}
    return (
        f"Global={100*m.get('global_aar'):.2f}% "
        f"CDR3={100*regs.get('CDR3'):.2f}%"
    )

lines = [
    "# epi_feat rerun vs previous @1500",
    "",
    "| Mode | Previous | Rerun | Abs diff (CA or Global%) |",
    "|---|---|---|---:|",
]
for mode in ("folding", "inverse_folding", "fr_folding", "fr_inverse_folding"):
    a, b = prev[mode], new[mode]
    if mode in ("folding", "fr_folding"):
        diff = abs(a["ca_rmsd"] - b["ca_rmsd"])
        lines.append(f"| {mode} | {fmt_fold(a)} | {fmt_fold(b)} | CA Δ={diff:.4f} |")
    else:
        diff = abs(a["global_aar"] - b["global_aar"]) * 100
        lines.append(f"| {mode} | {fmt_aar(a)} | {fmt_aar(b)} | Global Δ={diff:.3f}pp |")

out = Path("${OUT}")
out.mkdir(parents=True, exist_ok=True)
(out / "rerun_comparison.md").write_text("\\n".join(lines) + "\\n")
(out / "rerun_comparison.json").write_text(json.dumps({"previous": prev, "rerun": new}, indent=2))
print("\\n".join(lines))
print("wrote", out / "rerun_comparison.md")
PY

echo "[$(date)] done" | tee -a "$LOG"
