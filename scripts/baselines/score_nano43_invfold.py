#!/usr/bin/env python3
"""Score external invfold FASTA predictions on nano43 (Global/Fr/CDR AAR)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "smoke"))
sys.path.insert(0, str(REPO / "scripts"))

from eval_vhh_cdr_rmsd import to_lib_regions  # noqa: E402
from sab23_h2_nano_lib import compute_inverse_folding_metrics  # noqa: E402

BENCH_DEFAULT = REPO / "data-bin" / "benchmarks" / "mfdesign_nano_vhh"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-fasta", type=Path, required=True)
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DEFAULT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--ag-aware", action="store_true")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    native = {
        r.id: str(r.seq)
        for r in SeqIO.parse(str(args.bench_dir / "aatype.fasta"), "fasta")
    }
    cdr = to_lib_regions(
        json.loads((args.bench_dir / "cdr_regions_anarci_chothia.json").read_text())
    )
    rows = compute_inverse_folding_metrics(
        args.pred_fasta, native, cdr_regions=cdr
    )
    ok = [r for r in rows if r.get("seq_recovery") is not None]

    def mean_key(key: str):
        vals = [
            float(r[key])
            for r in ok
            if r.get(key) is not None and r[key] == r[key]
        ]
        return float(np.mean(vals)) if vals else None

    regions = {
        name: mean_key(f"{name}_seq_recovery")
        for name in ("Fr", "CDR1", "CDR2", "CDR3")
    }
    summary = {
        "tag": args.tag or args.out_dir.name,
        "n": len(ok),
        "n_pred": len(rows),
        "global_aar": mean_key("seq_recovery"),
        "regions": regions,
        "ag_aware": bool(args.ag_aware),
        "notes": args.notes,
        "pred_fasta": str(args.pred_fasta),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = sorted({k for r in rows for k in r})
    with (args.out_dir / "per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
