#!/usr/bin/env python3
"""Merge external invfold baseline summaries + DPLM2 reference into comparison.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "outputs" / "vhh_eval_ext_invfold_baselines"
DPLM_DEFAULT = (
    REPO
    / "outputs"
    / "vhh_eval_hybrid_epi_feat"
    / "fold50inv50_hybrid_epi_feat_step1500"
    / "summary.json"
)


def fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def row_from_summary(tag: str, s: dict) -> str:
    regs = s.get("regions") or {}
    return (
        f"| {tag} | {fmt_pct(s.get('global_aar'))} | {fmt_pct(regs.get('Fr'))} | "
        f"{fmt_pct(regs.get('CDR1'))} | {fmt_pct(regs.get('CDR2'))} | "
        f"{fmt_pct(regs.get('CDR3'))} | "
        f"{'yes' if s.get('ag_aware') else 'no'} | {s.get('notes','')} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--dplm-summary", type=Path, default=DPLM_DEFAULT)
    args = ap.parse_args()

    lines = [
        "# nano43 inverse folding — external baselines vs DPLM2",
        "",
        "Protocol A: Ab backbone (+ antigen if supported) → full Ab sequence; "
        "AAR vs native with ANARCI-Chothia CDR spans.",
        "",
        "| Method | Global | Fr | CDR1 | CDR2 | CDR3 | Ag-aware? | Notes |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    if args.dplm_summary.is_file():
        dplm = json.loads(args.dplm_summary.read_text())
        inv = dplm.get("inverse_folding") or dplm
        # reshape
        s = {
            "global_aar": inv.get("global_aar"),
            "regions": inv.get("regions") or {},
            "ag_aware": True,
            "notes": "DPLM2 fold50inv50_hybrid_epi_feat@1500",
        }
        lines.append(row_from_summary("DPLM2_hybrid_epi_feat@1500", s))

    skipped = args.out_root / "SKIPPED.md"
    for sub in sorted(args.out_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        summ = sub / "summary.json"
        if not summ.is_file():
            continue
        s = json.loads(summ.read_text())
        lines.append(row_from_summary(s.get("tag") or sub.name, s))

    if skipped.is_file():
        lines += ["", "## Skipped", "", skipped.read_text()]

    out = args.out_root / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
