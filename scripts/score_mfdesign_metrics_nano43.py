#!/usr/bin/env python3
"""Score nano43 folding predictions with MFDesign paper metrics.

Sec 4.2 (structure prediction / folding):
  - RMSD: whole-chain Cα superimpose, RMSD over all Cα
  - Loop-RMSD: whole-chain align, RMSD on CDR-H3 middle loop (Chothia: drop 2 aa each end)
  - TM-score: global fold similarity

Sec 4.1 style (co-design / Table 3, for reference):
  - CDR-H3 RMSD: framework align, full CDR-H3 Cα
  - Loop-RMSD: framework align, CDR-H3 middle loop

Example:
  python scripts/score_mfdesign_metrics_nano43.py \\
    --pred-pdb-dir outputs/vhh_eval_hybrid_epi_feat/.../folding/pdb
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "smoke"))

from eval_vhh_cdr_rmsd import to_lib_regions  # noqa: E402
from sab23_h2_nano_lib import (  # noqa: E402
    _ca_rmsd_after_align,
    _chain_feats,
    _pair_structure_to_reference,
    _region_indices_matched,
    calc_tm_score,
    superimpose,
)

BENCH_DEFAULT = REPO / "data-bin" / "benchmarks" / "mfdesign_nano_vhh"


def _loop_indices_from_cdr3(cdr3_idx: np.ndarray) -> np.ndarray | None:
    """MFDesign Chothia: exclude first 2 and last 2 residues of CDR-H3."""
    if cdr3_idx is None or len(cdr3_idx) < 5:
        return None
    return cdr3_idx[2:-2]


def _mean_metric(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get("status") == "ok" and r.get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def compute_mfdesign_row(
    pred_pdb: Path,
    gt_pdb: Path,
    cdr_regions: dict | None,
) -> dict:
    sid = pred_pdb.stem
    row: dict = {"sample_id": sid}
    if not gt_pdb.is_file():
        return {**row, "status": "missing_gt"}

    try:
        paired = _pair_structure_to_reference(_chain_feats(pred_pdb), _chain_feats(gt_pdb))
        if paired is None:
            return {**row, "status": "align_failed"}
        if paired["seq_identity"] < 0.95:
            return {**row, "status": f"low_seq_identity:{paired['seq_identity']:.2f}"}

        pred_ca = paired["pred_ca"]
        gt_ca = paired["gt_ca"]
        n = paired["n_aligned"]
        all_idx = np.arange(n)
        mask = torch.ones(n, dtype=torch.bool)

        _, tm = calc_tm_score(
            paired["pred_bb_n3"], paired["gt_bb_n3"], paired["pred_seq"], paired["gt_seq"]
        )

        fold_rmsd = _ca_rmsd_after_align(pred_ca, gt_ca, all_idx, align_indices=all_idx)
        fold_loop = float("nan")
        codesign_cdr3 = float("nan")
        codesign_loop = float("nan")

        if cdr_regions and sid in cdr_regions:
            reg = cdr_regions[sid]
            gt_idx = paired["gt_idx"]
            fr_idx = _region_indices_matched(reg, "Fr", gt_idx, None, "sequence_0based")
            cdr3_idx = _region_indices_matched(reg, "CDR3", gt_idx, None, "sequence_0based")
            loop_idx = _loop_indices_from_cdr3(cdr3_idx)

            if loop_idx is not None and len(loop_idx) > 0:
                fold_loop = _ca_rmsd_after_align(
                    pred_ca, gt_ca, loop_idx, align_indices=all_idx
                )
            if fr_idx is not None and len(fr_idx) >= 3 and cdr3_idx is not None and len(cdr3_idx):
                codesign_cdr3 = _ca_rmsd_after_align(
                    pred_ca, gt_ca, cdr3_idx, align_indices=fr_idx
                )
            if fr_idx is not None and len(fr_idx) >= 3 and loop_idx is not None and len(loop_idx):
                codesign_loop = _ca_rmsd_after_align(
                    pred_ca, gt_ca, loop_idx, align_indices=fr_idx
                )

        return {
            **row,
            "status": "ok",
            "fold_rmsd": fold_rmsd,
            "fold_loop_rmsd": fold_loop,
            "codesign_cdr3_rmsd": codesign_cdr3,
            "codesign_loop_rmsd": codesign_loop,
            "tm_score": float(tm),
            "length": n,
        }
    except Exception as exc:
        return {**row, "status": f"error:{exc}"}


def score_dir(pred_pdb_dir: Path, gt_pdb_dir: Path, cdr_regions: dict | None) -> dict:
    rows = [
        compute_mfdesign_row(p, gt_pdb_dir / f"{p.stem}.pdb", cdr_regions)
        for p in sorted(pred_pdb_dir.glob("*.pdb"))
    ]
    ok = [r for r in rows if r.get("status") == "ok"]
    return {
        "n_pred": len(rows),
        "n_ok": len(ok),
        "fold_rmsd": _mean_metric(ok, "fold_rmsd"),
        "fold_loop_rmsd": _mean_metric(ok, "fold_loop_rmsd"),
        "codesign_cdr3_rmsd": _mean_metric(ok, "codesign_cdr3_rmsd"),
        "codesign_loop_rmsd": _mean_metric(ok, "codesign_loop_rmsd"),
        "tm_score": _mean_metric(ok, "tm_score"),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MFDesign-style nano43 folding metrics")
    ap.add_argument("--pred-pdb-dir", type=Path, required=True)
    ap.add_argument(
        "--gt-pdb-dir",
        type=Path,
        default=BENCH_DEFAULT / "pdb_h",
    )
    ap.add_argument(
        "--cdr-json",
        type=Path,
        default=BENCH_DEFAULT / "cdr_regions_anarci_chothia.json",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    cdr_raw = json.loads(args.cdr_json.read_text())
    cdr = to_lib_regions(cdr_raw)

    result = score_dir(args.pred_pdb_dir, args.gt_pdb_dir, cdr)
    summary = {k: v for k, v in result.items() if k != "rows"}

    print(json.dumps(summary, indent=2))
    print()
    print("MFDesign folding (Sec 4.2) — compare to Table 4:")
    print(f"  RMSD       = {summary['fold_rmsd']:.2f} Å")
    print(f"  Loop-RMSD  = {summary['fold_loop_rmsd']:.2f} Å")
    print(f"  TM-score   = {summary['tm_score']:.3f}")
    print()
    print("MFDesign co-design style (Sec 4.1 / Table 3 nanobody reference):")
    print(f"  CDR-H3 RMSD = {summary['codesign_cdr3_rmsd']:.2f} Å")
    print(f"  Loop-RMSD   = {summary['codesign_loop_rmsd']:.2f} Å")

    out_json = args.out_json or args.pred_pdb_dir.parent / "mfdesign_metrics.json"
    out_csv = args.out_csv or args.pred_pdb_dir.parent / "mfdesign_metrics.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({**summary, "pred_pdb_dir": str(args.pred_pdb_dir)}, indent=2) + "\n")

    fields = [
        "sample_id", "status", "fold_rmsd", "fold_loop_rmsd",
        "codesign_cdr3_rmsd", "codesign_loop_rmsd", "tm_score", "length",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in result["rows"]:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
