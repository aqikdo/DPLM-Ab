#!/usr/bin/env python3
"""Prepare nano43 H+Ag complex PDBs and chain table for external invfold baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Bio import SeqIO

REPO = Path(__file__).resolve().parents[2]
BENCH_DEFAULT = REPO / "data-bin" / "benchmarks" / "mfdesign_nano_vhh"
OUT_DEFAULT = REPO / "outputs" / "vhh_eval_ext_invfold_baselines" / "_inputs"


def _extract_chains(src_pdb: Path, dst_pdb: Path, chain_ids: list[str]) -> None:
    keep = set(chain_ids)
    lines_out = []
    with src_pdb.open() as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[21] in keep:
                    lines_out.append(line)
            elif line.startswith("END"):
                break
    lines_out.append("END\n")
    dst_pdb.parent.mkdir(parents=True, exist_ok=True)
    dst_pdb.write_text("".join(lines_out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--ab-only",
        action="store_true",
        help="Use heavy-chain-only PDBs (no antigen in input).",
    )
    args = ap.parse_args()

    rows = json.loads((args.bench_dir / "antigen_eval_rows.json").read_text())
    native = {
        r.id: str(r.seq)
        for r in SeqIO.parse(str(args.bench_dir / "aatype.fasta"), "fasta")
    }

    pdb_h_dir = args.bench_dir / "pdb_h"
    pdb_dir = args.out_dir / ("ab_only_pdbs" if args.ab_only else "complex_pdbs")
    pdb_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for row in rows:
        sid = row["sample_id"]
        h = str(row["chain_id"])
        ag = row["antigen_chain"]
        if isinstance(ag, list):
            ag = str(ag[0])
        else:
            ag = str(ag)
        if args.ab_only:
            src = pdb_h_dir / f"{sid}.pdb"
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = pdb_dir / f"{sid}.pdb"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            ag_chains: list[str] = []
        else:
            src = Path(row["pdb_path"])
            if not src.is_file():
                raise FileNotFoundError(src)
            # antigen_chain may be concatenated IDs, e.g. "BD"
            present = set()
            with src.open() as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        present.add(line[21])
            ag_chains = [ag] if ag in present else [c for c in str(ag) if c in present]
            keep = [h] + [c for c in ag_chains if c != h]
            dst = pdb_dir / f"{sid}.pdb"
            _extract_chains(src, dst, keep)
        aa = native.get(sid) or row.get("aa_seq")
        table.append(
            {
                "sample_id": sid,
                "entry_id": row["entry_id"],
                "heavy_chain": h,
                "antigen_chain": ag,
                "antigen_chains": ag_chains,
                "ab_only": args.ab_only,
                "pdb_path": str(dst),
                "src_pdb": str(src),
                "length": len(aa) if aa else row.get("length"),
                "aa_seq": aa,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "samples.json").write_text(json.dumps(table, indent=2))
    lines = ["sample_id\theavy\tantigen\tpdb"]
    for t in table:
        lines.append(
            f"{t['sample_id']}\t{t['heavy_chain']}\t{t['antigen_chain']}\t{t['pdb_path']}"
        )
    (args.out_dir / "samples.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(table)} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
