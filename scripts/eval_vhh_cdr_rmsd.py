#!/usr/bin/env python3
"""Compute framework-aligned CDR1/2/3 CA RMSD on existing VHH folding predictions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from anarci import run_anarci

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "smoke"))

from sab23_h2_nano_lib import (  # noqa: E402
    REGION_NAMES,
    compute_folding_metrics,
    summarize_region_metrics,
)

# Chothia VH CDR residue numbers (inclusive)
CHOTHIA_CDR1 = range(26, 33)
CHOTHIA_CDR2 = range(52, 57)
# CDR3: 95–102 (including lettered insertions e.g. 100A)


def _anarci_seq_to_chothia(seq: str) -> list[tuple[int, str]] | None:
    """Return list of (chothia_number, aa) for non-gap residues in query order."""
    res = run_anarci([("q", seq)], scheme="chothia", output=False)
    numbering = res[1][0]
    if not numbering:
        return None
    # Prefer first heavy-chain-like domain
    domain = numbering[0]
    numbered = domain[0]  # list of ((num, insertion), aa)
    out: list[tuple[int, str]] = []
    for (num, _ins), aa in numbered:
        if aa == "-":
            continue
        out.append((int(num), aa))
    return out if out else None


def _cdr_spans_from_chothia_map(chothia_map: list[tuple[int, str]]) -> dict | None:
    """Build inclusive 0-based sequence spans for CDR1/2/3 from Chothia map."""
    cdr1_idx = [i for i, (n, _) in enumerate(chothia_map) if n in CHOTHIA_CDR1]
    cdr2_idx = [i for i, (n, _) in enumerate(chothia_map) if n in CHOTHIA_CDR2]
    cdr3_idx = [i for i, (n, _) in enumerate(chothia_map) if 95 <= n <= 102]
    if not cdr1_idx or not cdr2_idx or not cdr3_idx:
        return None
    return {
        "CDR1": (min(cdr1_idx), max(cdr1_idx)),
        "CDR2": (min(cdr2_idx), max(cdr2_idx)),
        "CDR3": (min(cdr3_idx), max(cdr3_idx)),
        "length": len(chothia_map),
    }


def _pdb_sequence(pdb_path: Path) -> str | None:
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    chains = list(structure[0].get_chains())
    if not chains:
        return None
    aa3to1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    seq = []
    for res in chains[0]:
        if res.id[0] != " ":
            continue
        aa = aa3to1.get(res.get_resname())
        if aa:
            seq.append(aa)
    return "".join(seq) if seq else None


def build_cdr_regions_for_benchmark(gt_pdb_dir: Path) -> dict[str, dict]:
    regions: dict[str, dict] = {}
    failed = 0
    substring = 0
    for pdb in sorted(gt_pdb_dir.glob("*.pdb")):
        sid = pdb.stem
        seq = _pdb_sequence(pdb)
        if not seq:
            failed += 1
            continue
        cmap = _anarci_seq_to_chothia(seq)
        if not cmap:
            failed += 1
            continue
        mapped_seq = "".join(aa for _, aa in cmap)
        offset = 0
        if mapped_seq != seq:
            # Fab / multi-domain: ANARCI returns the VH domain as a substring
            idx = seq.find(mapped_seq)
            if idx < 0:
                failed += 1
                continue
            offset = idx
            substring += 1
        spans = _cdr_spans_from_chothia_map(cmap)
        if not spans:
            failed += 1
            continue
        regions[sid] = {
            "cdr1": [spans["CDR1"][0] + offset, spans["CDR1"][1] + offset],
            "cdr2": [spans["CDR2"][0] + offset, spans["CDR2"][1] + offset],
            "cdr3": [spans["CDR3"][0] + offset, spans["CDR3"][1] + offset],
            "length": len(seq),
            "vh_offset": offset,
            "vh_len": len(mapped_seq),
        }
    print(
        f"[cdr] {gt_pdb_dir.parent.name}: annotated {len(regions)} "
        f"(substring VH {substring}), failed {failed}"
    )
    return regions

def to_lib_regions(raw: dict[str, dict]) -> dict[str, dict]:
    """Include a synthetic VH-only framework span used by custom Fr RMSD.

    CDR1/2/3 are inclusive 0-based indices into the native PDB sequence.
    We also stash vh_start/vh_end so Fr = VH \\ CDRs (not the whole chain).
    """
    out = {}
    for sid, reg in raw.items():
        vh0 = int(reg.get("vh_offset", 0))
        vh1 = vh0 + int(reg.get("vh_len", reg["length"])) - 1
        out[sid] = {
            "CDR1": tuple(reg["cdr1"]),
            "CDR2": tuple(reg["cdr2"]),
            "CDR3": tuple(reg["cdr3"]),
            "length": reg["length"],
            "vh_start": vh0,
            "vh_end": vh1,
        }
    return out


def _vh_framework_indices(reg: dict, gt_idx) -> "np.ndarray | None":
    """Paired-row indices that are in the VH framework (not CDRs)."""
    import numpy as np

    vh0, vh1 = reg["vh_start"], reg["vh_end"]
    cdr_mask = np.zeros(len(gt_idx), dtype=bool)
    for key in ("CDR1", "CDR2", "CDR3"):
        start, end = reg[key]
        cdr_mask |= (gt_idx >= start) & (gt_idx <= end)
    in_vh = (gt_idx >= vh0) & (gt_idx <= vh1)
    idx = np.where(in_vh & ~cdr_mask)[0]
    return idx if len(idx) else None


def eval_one(pred_dir: Path, gt_dir: Path, cdr_regions: dict) -> dict:
    """Compute metrics then replace Fr RMSD with VH-framework-only alignment."""
    from sab23_h2_nano_lib import (
        _ca_rmsd_after_align,
        _chain_feats,
        _pair_structure_to_reference,
        _region_indices_matched,
    )

    rows = compute_folding_metrics(
        pred_dir,
        gt_dir,
        cdr_regions=cdr_regions,
        cdr_numbering="sequence_0based",
    )
    # Recompute regional RMSD with VH-only framework alignment
    fixed = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        sid = row["sample_id"]
        if sid not in cdr_regions:
            continue
        reg = cdr_regions[sid]
        pred_pdb = pred_dir / f"{sid}.pdb"
        gt_pdb = gt_dir / f"{sid}.pdb"
        try:
            paired = _pair_structure_to_reference(_chain_feats(pred_pdb), _chain_feats(gt_pdb))
            if paired is None or paired["seq_identity"] < 0.95:
                continue
            pred_ca, gt_ca = paired["pred_ca"], paired["gt_ca"]
            fr_idx = _vh_framework_indices(reg, paired["gt_idx"])
            if fr_idx is None or len(fr_idx) < 3:
                continue
            row = dict(row)
            row["Fr_ca_rmsd"] = _ca_rmsd_after_align(
                pred_ca, gt_ca, fr_idx, align_indices=fr_idx
            )
            for region in ("CDR1", "CDR2", "CDR3"):
                idx = _region_indices_matched(
                    reg, region, paired["gt_idx"], None, "sequence_0based"
                )
                if idx is None or len(idx) == 0:
                    row[f"{region}_ca_rmsd"] = float("nan")
                else:
                    row[f"{region}_ca_rmsd"] = _ca_rmsd_after_align(
                        pred_ca, gt_ca, idx, align_indices=fr_idx
                    )
            fixed.append(row)
        except Exception:
            continue

    region_sum = summarize_region_metrics(fixed, "ca_rmsd")
    return {
        "n_pred": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_with_cdr": len(fixed),
        "regions_ca_rmsd": region_sum,
        "rows": fixed,
    }


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return True


def main() -> None:
    benchmarks = {
        "pxmeter_ab": REPO / "data-bin/benchmarks/pxmeter_ab_vhh",
        "af3_ab": REPO / "data-bin/benchmarks/af3_ab_vhh",
    }
    models = [
        ("pretrained", None, REPO / "outputs/vhh_fold_eval/pre"),
        ("sr000@500", 0.00, REPO / "outputs/vhh_fold_eval_mixed_mmseqs_clean_sr000_by_step/step_500"),
        ("sr005@500", 0.05, REPO / "outputs/vhh_fold_eval_mixed_mmseqs_clean_sr005_by_step/step_500"),
        ("sr010@500", 0.10, REPO / "outputs/vhh_fold_eval_mixed_mmseqs_clean_sr010_by_step/step_500"),
        ("sr025@500", 0.25, REPO / "outputs/vhh_fold_eval_mixed_mmseqs_clean_by_step/step_500"),
    ]

    out_dir = REPO / "outputs/vhh_eval_mixed_mmseqs_clean_4way"
    out_dir.mkdir(parents=True, exist_ok=True)

    cdr_cache = {}
    for bench, bench_dir in benchmarks.items():
        raw = build_cdr_regions_for_benchmark(bench_dir / "pdb_h")
        cache_path = bench_dir / "cdr_regions_anarci_chothia.json"
        cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        cdr_cache[bench] = to_lib_regions(raw)

    results = []
    lines = [
        "# Framework-aligned CDR loop CA RMSD (Å)\n",
        "Protocol: ANARCI Chothia CDR1/2/3 on native PDB sequence "
        "(Fab chains: use VH domain substring); "
        "superimpose on **VH framework** CA, then RMSD on each CDR.\n",
        "N = chains successfully annotated as Ig VH by ANARCI "
        "(antigens / failed HMMER are excluded). "
        "Models: pretrained DPLM2-650M + mixed LoRA @ step 500.\n",
        "## Mean CA RMSD by region\n",
        "| Model | Bench | N | Fr | CDR1 | CDR2 | CDR3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for tag, ratio, root in models:
        for bench, bench_dir in benchmarks.items():
            pred = root / bench / "folding" / "pdb"
            gt = bench_dir / "pdb_h"
            if not pred.is_dir():
                print(f"[skip] missing {pred}")
                continue
            print(f"[eval] {tag} {bench}", flush=True)
            res = eval_one(pred, gt, cdr_cache[bench])
            reg = res["regions_ca_rmsd"]

            def m(name: str) -> str:
                if name not in reg:
                    return "—"
                return f"{reg[name]['mean']:.2f}"

            lines.append(
                f"| {tag} | {bench} | {res['n_with_cdr']} | "
                f"{m('Fr')} | {m('CDR1')} | {m('CDR2')} | {m('CDR3')} |"
            )
            results.append(
                {
                    "tag": tag,
                    "ratio": ratio,
                    "bench": bench,
                    "n_ok": res["n_ok"],
                    "n_with_cdr": res["n_with_cdr"],
                    "regions_ca_rmsd": {
                        k: {kk: vv for kk, vv in v.items() if kk in ("count", "mean", "median")}
                        for k, v in reg.items()
                    },
                }
            )
            # write per-model region CSV
            csv_path = out_dir / f"cdr_rmsd_{tag.replace('@','_')}_{bench}.csv"
            if res["rows"]:
                keys = ["sample_id", "bb_tmscore", "ca_rmsd"] + [
                    f"{r}_ca_rmsd" for r in REGION_NAMES
                ]
                with csv_path.open("w", newline="", encoding="utf-8") as fp:
                    w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
                    w.writeheader()
                    for row in res["rows"]:
                        w.writerow(row)

    # Compact PXM / AF3 side-by-side for CDR3
    lines += [
        "",
        "## Compact mean CA RMSD (Å)\n",
        "| Model | PXM Fr | PXM CDR1 | PXM CDR2 | PXM CDR3 | AF3 Fr | AF3 CDR1 | AF3 CDR2 | AF3 CDR3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by = {(r["tag"], r["bench"]): r for r in results}

    def cell(tag: str, bench: str, region: str) -> str:
        r = by.get((tag, bench))
        if not r:
            return "—"
        reg = r["regions_ca_rmsd"].get(region)
        return f"{reg['mean']:.2f}" if reg else "—"

    for tag, _, _ in models:
        lines.append(
            "| {tag} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} |".format(
                tag=tag,
                a=cell(tag, "pxmeter_ab", "Fr"),
                b=cell(tag, "pxmeter_ab", "CDR1"),
                c=cell(tag, "pxmeter_ab", "CDR2"),
                d=cell(tag, "pxmeter_ab", "CDR3"),
                e=cell(tag, "af3_ab", "Fr"),
                f=cell(tag, "af3_ab", "CDR1"),
                g=cell(tag, "af3_ab", "CDR2"),
                h=cell(tag, "af3_ab", "CDR3"),
            )
        )

    # deltas vs pretrained on CDR3
    lines += [
        "",
        "## Δ CDR3 CA RMSD vs pretrained (negative = better)\n",
        "| Model | Δ PXM CDR3 | Δ AF3 CDR3 |",
        "|---|---:|---:|",
    ]
    for tag, _, _ in models:
        if tag == "pretrained":
            continue
        parts = []
        for bench in ("pxmeter_ab", "af3_ab"):
            pre = by[("pretrained", bench)]["regions_ca_rmsd"].get("CDR3", {}).get("mean")
            cur = by[(tag, bench)]["regions_ca_rmsd"].get("CDR3", {}).get("mean")
            if pre is None or cur is None:
                parts.append("—")
            else:
                parts.append(f"{cur - pre:+.2f}")
        lines.append(f"| {tag} | {parts[0]} | {parts[1]} |")

    md_path = out_dir / "cdr_rmsd_step500_vs_pretrained.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "cdr_rmsd_step500_vs_pretrained.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"[DONE] {md_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
