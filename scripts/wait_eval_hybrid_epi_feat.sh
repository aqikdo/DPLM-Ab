#!/usr/bin/env bash
# Wait for fold50inv50_hybrid_epi_feat step_1500, then 4-mode eval vs hybrid epicrop.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYTHON="${PYTHON:-/home/zhaohongyan/miniconda3/envs/dplm/bin/python}"
CKPT_NEW="logs/fold50inv50_hybrid_epi_feat/checkpoints/every250/step_1500.ckpt"
CKPT_BASE="logs/fold50inv50_hybrid_noise_epi01/checkpoints/every250/step_1500.ckpt"
OUT="outputs/vhh_eval_hybrid_epi_feat"
LOG="logs/eval_hybrid_epi_feat.log"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

echo "[$(date)] waiting for ${CKPT_NEW}" | tee -a "$LOG"
while [[ ! -f "$CKPT_NEW" ]]; do sleep 60; done
sleep 30
echo "[$(date)] found ckpt; eval on GPU ${GPU}" | tee -a "$LOG"

mkdir -p "$OUT"
# Reuse hybrid epicrop summary if present; only eval new model
BASE_SUM="outputs/vhh_eval_hybrid_epi_xattn/fold50inv50_hybrid_noise_step1500_epicrop/summary.json"
env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" -u scripts/eval_antigen_cond_nano43.py \
  --out-root "$OUT" \
  --antigen-max-len 256 \
  --modes folding,inverse_folding,fr_folding,fr_inverse_folding \
  --ckpts \
    "fold50inv50_hybrid_epi_feat_step1500=${CKPT_NEW}" \
  >> "$LOG" 2>&1

"${PYTHON}" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

def pct(x):
    return f"{100.0 * x:.1f}" if x is not None else "—"

out = Path("outputs/vhh_eval_hybrid_epi_feat")
by_tag = {}
# fair hybrid baseline (epitope-aware crop, no feature)
base_sj = Path("outputs/vhh_eval_hybrid_epi_xattn/fold50inv50_hybrid_noise_step1500_epicrop/summary.json")
if base_sj.is_file():
    s = json.loads(base_sj.read_text())
    s["tag"] = "fold50inv50_hybrid_noise_step1500_epicrop"
    by_tag[s["tag"]] = s
# wrong hard-mask run for reference
hard_sj = Path("outputs/vhh_eval_hybrid_epi_xattn/fold50inv50_hybrid_epi_xattn_step1500/summary.json")
if hard_sj.is_file():
    s = json.loads(hard_sj.read_text())
    s["tag"] = "fold50inv50_hybrid_epi_xattn_step1500 (hard-mask WRONG)"
    by_tag[s["tag"]] = s
for sj in sorted(out.glob("*/summary.json")):
    s = json.loads(sj.read_text())
    s["tag"] = sj.parent.name
    by_tag[s["tag"]] = s

order = [
    "fold50inv50_hybrid_noise_step1500_epicrop",
    "fold50inv50_hybrid_epi_xattn_step1500 (hard-mask WRONG)",
    "fold50inv50_hybrid_epi_feat_step1500",
]
summaries = [by_tag[t] for t in order if t in by_tag]
modes = ["folding", "inverse_folding", "fr_folding", "fr_inverse_folding"]
lines = [
    "# Epitope 0/1 feature (additive) vs hybrid / hard-mask (nano43 @1500)",
    "",
    "- Correct: `epi_feat` = learned Embedding(2) added to Ag encoder states for cross-attn; **full Ag still attended**.",
    "- Baseline: hybrid with epitope-aware crop (no epitope feature).",
    "- Wrong prior: hard key mask (non-epitope dropped).",
    "",
]
for mode in modes:
    if mode in ("folding", "fr_folding"):
        lines += [
            f"## {mode}", "",
            "| Model | Mean CA RMSD (Å) | Mean TM | Fr RMSD | CDR1 | CDR2 | **CDR3** | N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for s in summaries:
            m = s.get(mode) or {}
            regs = m.get("regions_ca_rmsd") or {}
            ca, tm = m.get("ca_rmsd"), m.get("bb_tmscore")
            lines.append(
                "| {tag} | {ca} | {tm} | {fr} | {c1} | {c2} | {c3} | {n} |".format(
                    tag=s["tag"],
                    ca=f"{ca:.2f}" if ca is not None else "—",
                    tm=f"{tm:.3f}" if tm is not None else "—",
                    fr=f"{regs['Fr']:.2f}" if regs.get("Fr") is not None else "—",
                    c1=f"{regs['CDR1']:.2f}" if regs.get("CDR1") is not None else "—",
                    c2=f"{regs['CDR2']:.2f}" if regs.get("CDR2") is not None else "—",
                    c3=f"{regs['CDR3']:.2f}" if regs.get("CDR3") is not None else "—",
                    n=m.get("n") or "—",
                )
            )
        lines.append("")
    else:
        lines += [
            f"## {mode}", "",
            "| Model | Global AAR % | Fr % | CDR1 % | CDR2 % | **CDR3 %** | N |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in summaries:
            m = s.get(mode) or {}
            regs = m.get("regions") or {}
            lines.append(
                "| {tag} | {g} | {fr} | {c1} | {c2} | {c3} | {n} |".format(
                    tag=s["tag"],
                    g=pct(m.get("global_aar")),
                    fr=pct(regs.get("Fr")),
                    c1=pct(regs.get("CDR1")),
                    c2=pct(regs.get("CDR2")),
                    c3=pct(regs.get("CDR3")),
                    n=m.get("n") or "—",
                )
            )
        lines.append("")

out.mkdir(parents=True, exist_ok=True)
(out / "comparison.md").write_text("\n".join(lines) + "\n")
(out / "comparison.json").write_text(json.dumps(summaries, indent=2, default=str))
print("wrote", out / "comparison.md")
PY

echo "[$(date)] done" | tee -a "$LOG"
