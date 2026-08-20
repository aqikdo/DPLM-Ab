#!/usr/bin/env python3
"""Run ProteinMPNN / AbMPNN on nano43: design heavy chain, fix antigen."""

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
PYTHON = os.environ.get("PYTHON", sys.executable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-json", type=Path, required=True)
    ap.add_argument("--pmpnn-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--path-to-model-weights", type=Path, required=True)
    ap.add_argument("--model-name", type=str, default="v_48_020")
    ap.add_argument("--sampling-temp", type=str, default="0.2")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--num-seq-per-target", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = json.loads(args.samples_json.read_text())
    if args.limit:
        samples = samples[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = args.out_dir / "pdbs"
    pdb_dir.mkdir(exist_ok=True)
    for s in samples:
        dst = pdb_dir / f"{s['sample_id']}.pdb"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(Path(s["pdb_path"]).resolve())

    parsed = args.out_dir / "parsed_pdbs.jsonl"
    parse_py = args.pmpnn_dir / "helper_scripts" / "parse_multiple_chains.py"
    run_py = args.pmpnn_dir / "protein_mpnn_run.py"

    subprocess.check_call(
        [
            PYTHON,
            str(parse_py),
            "--input_path",
            str(pdb_dir),
            "--output_path",
            str(parsed),
        ]
    )

    # Build chain assignment: design heavy, fix antigen (and any other chains)
    with parsed.open() as f:
        parsed_rows = [json.loads(line) for line in f if line.strip()]
    by_name = {r["name"]: r for r in parsed_rows}
    assigned = {}
    for s in samples:
        name = s["sample_id"]
        row = by_name[name]
        all_chains = [k[-1] for k in row if k.startswith("seq_chain")]
        designed = [s["heavy_chain"]]
        fixed = [c for c in all_chains if c not in designed]
        assigned[name] = (designed, fixed)
    chain_jsonl = args.out_dir / "assigned_chains.jsonl"
    chain_jsonl.write_text(json.dumps(assigned) + "\n")

    weights_dir = str(args.path_to_model_weights)
    if not weights_dir.endswith("/"):
        weights_dir += "/"

    cmd = [
        PYTHON,
        str(run_py.resolve()),
        "--jsonl_path",
        str(parsed.resolve()),
        "--chain_id_jsonl",
        str(chain_jsonl.resolve()),
        "--out_folder",
        str(args.out_dir.resolve()),
        "--num_seq_per_target",
        str(args.num_seq_per_target),
        "--sampling_temp",
        args.sampling_temp,
        "--seed",
        str(args.seed),
        "--batch_size",
        "1",
        "--path_to_model_weights",
        str(Path(weights_dir).resolve()) + "/",
        "--model_name",
        args.model_name,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(args.pmpnn_dir.resolve()))

    seqs_dir = args.out_dir / "seqs"
    pred_records = []
    for s in samples:
        fa = seqs_dir / f"{s['sample_id']}.fa"
        if not fa.exists():
            print(f"MISSING {fa}", flush=True)
            continue
        recs = list(SeqIO.parse(str(fa), "fasta"))
        designed = None
        for r in recs:
            if "sample=" in r.description:
                designed = str(r.seq)
                break
        if designed is None and len(recs) >= 2:
            designed = str(recs[1].seq)
        if designed is None:
            continue
        # Output is designed chains only concatenated in designed order (heavy only)
        L = int(s["length"])
        if len(designed) >= L:
            # When only H is designed, sequence should be H length.
            # If both chains somehow present, take designed chain portion.
            seq = designed if len(designed) == L else designed[:L]
        else:
            seq = designed
        pred_records.append(SeqRecord(Seq(seq), id=s["sample_id"], description=""))

    out_fa = args.out_dir / "predictions.fasta"
    SeqIO.write(pred_records, str(out_fa), "fasta")
    print(f"wrote {len(pred_records)} seqs -> {out_fa}")


if __name__ == "__main__":
    main()
