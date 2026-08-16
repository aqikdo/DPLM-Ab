#!/usr/bin/env python3
"""Merge hybrid-noise eval summaries with baselines into comparison.md."""
from __future__ import annotations

import json
from pathlib import Path


def pct(x):
    return f"{100.0 * x:.1f}" if x is not None else "—"


def main():
    out = Path("outputs/vhh_eval_hybrid_noise_vs_baselines")
    by_tag = {}
    for sj in sorted(out.glob("*/summary.json")):
        s = json.loads(sj.read_text())
        s["tag"] = sj.parent.name
        by_tag[s["tag"]] = s

    order = [
        "fold_B_step1500",
        "fold50inv50_step1500",
        "fold_B_hybrid_noise_step1500",
        "fold50inv50_hybrid_noise_step1500",
    ]
    all_summaries = [by_tag[t] for t in order if t in by_tag]
    modes = ["folding", "inverse_folding", "fr_folding", "fr_inverse_folding"]
    lines = [
        "# Hybrid noise (50% original + 50% progressive CDR) vs baselines",
        "",
        "Progressive CDR half: mask_scale 0.1→1.0 by step **1500**.",
        "",
    ]
    for mode in modes:
        if mode in ("folding", "fr_folding"):
            lines += [
                f"## {mode}",
                "",
                "| Model | Mean CA RMSD (Å) | Mean TM | Fr RMSD | CDR1 | CDR2 | **CDR3** | N |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for s in all_summaries:
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
                f"## {mode}",
                "",
                "| Model | Global AAR % | Fr % | CDR1 % | CDR2 % | **CDR3 %** | N |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for s in all_summaries:
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

    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    (out / "comparison.json").write_text(json.dumps(all_summaries, indent=2) + "\n")
    print((out / "comparison.md").read_text())


if __name__ == "__main__":
    main()
