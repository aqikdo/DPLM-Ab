#!/usr/bin/env python3
"""Run AntiFold or ESM-IF1 on nano43 (nanobody + antigen). Use logits argmax for AAR."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

REPO = Path(__file__).resolve().parents[2]
ANTIFOLD_ROOT = REPO / "third_party" / "AntiFold"
PYTHON = os.environ.get("PYTHON", sys.executable)
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def seq_from_logits_csv(csv_path: Path, heavy_chain: str) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "pdb_chain" in df.columns:
        df = df[df["pdb_chain"].astype(str) == str(heavy_chain)]
    if "top_res" in df.columns:
        return "".join(str(x) for x in df["top_res"].tolist())
    aa_cols = [c for c in df.columns if c in AA20]
    seq = []
    for _, row in df.iterrows():
        best = max(aa_cols, key=lambda c: float(row[c]))
        seq.append(best)
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--esm-if1-mode", action="store_true")
    ap.add_argument(
        "--ab-only",
        action="store_true",
        help="Design nanobody from single-chain PDB (no antigen context).",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = json.loads(args.samples_json.read_text())
    if args.limit:
        samples = samples[: args.limit]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ANTIFOLD_ROOT) + (
        (":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )

    pred_records = []
    for s in samples:
        sid = s["sample_id"]
        out_s = (work / sid).resolve()
        out_s.mkdir(exist_ok=True)
        existing = [
            p
            for p in out_s.rglob("*.csv")
            if "log" not in p.name.lower() and p.name != "log.txt"
        ]
        if existing:
            seq = seq_from_logits_csv(existing[0], s["heavy_chain"])
            L = int(s["length"])
            if len(seq) != L:
                print(f"WARN {sid} pred_len={len(seq)} native_len={L}", flush=True)
            pred_records.append(SeqRecord(Seq(seq), id=sid, description=""))
            print(f"SKIP {sid} (existing csv)", flush=True)
            continue
        ab_only = args.ab_only or s.get("ab_only")
        pdb_path = Path(s["pdb_path"])
        if not pdb_path.is_absolute():
            pdb_path = (REPO / pdb_path).resolve()
        cmd = [
            PYTHON,
            str(ANTIFOLD_ROOT / "antifold" / "main.py"),
            "--pdb_file",
            str(pdb_path),
            "--nanobody_chain",
            s["heavy_chain"],
            "--out_dir",
            str(out_s),
            "--verbose",
            "0",
        ]
        if not ab_only:
            ag_chains = s.get("antigen_chains") or []
            ag = s.get("antigen_chain")
            # AntiFold Agchain is a single chain id; expand "BD" via antigen_chains.
            if ag_chains:
                ag_id = str(ag_chains[0])
            elif ag and len(str(ag)) == 1:
                ag_id = str(ag)
            elif ag:
                # fallback: first char of concatenated antigen ids
                ag_id = str(ag)[0]
            else:
                ag_id = None
            if ag_id:
                cmd.extend(["--antigen_chain", ag_id])
                if ag_chains and len(ag_chains) > 1:
                    print(
                        f"NOTE {sid}: multi Ag {ag_chains}; AntiFold using first={ag_id}",
                        flush=True,
                    )
        if args.esm_if1_mode:
            cmd.append("--esm_if1_mode")
        print("RUN", sid, " ".join(cmd[-8:]), flush=True)
        r = subprocess.run(
            cmd, cwd=str(ANTIFOLD_ROOT), env=env, capture_output=True, text=True
        )
        csvs = [
            p
            for p in out_s.rglob("*.csv")
            if "log" not in p.name.lower() and p.name != "log.txt"
        ]
        if r.returncode != 0 or not csvs:
            print(r.stdout[-4000:] if r.stdout else "", flush=True)
            print(r.stderr[-4000:] if r.stderr else "", flush=True)
            raise SystemExit(
                f"AntiFold failed on {sid} rc={r.returncode} csvs={len(csvs)}"
            )
        seq = seq_from_logits_csv(csvs[0], s["heavy_chain"])
        L = int(s["length"])
        if len(seq) != L:
            print(f"WARN {sid} pred_len={len(seq)} native_len={L}", flush=True)
        pred_records.append(SeqRecord(Seq(seq), id=sid, description=""))

    out_fa = out_dir / "predictions.fasta"
    SeqIO.write(pred_records, str(out_fa), "fasta")
    print(f"wrote {len(pred_records)} -> {out_fa}")


if __name__ == "__main__":
    main()
