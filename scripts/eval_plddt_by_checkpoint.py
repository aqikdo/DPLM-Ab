#!/usr/bin/env python3
"""Batch-eval mean pLDDT (sequence_generation -> ESMFold) for finetune checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def step_from_ckpt(path: Path) -> int:
    m = re.search(r"step_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def clean_aa_sequence(seq: str) -> str | None:
    """Keep the trailing VHH-like amino-acid run (sequence_generation may prefix junk)."""
    seq = re.sub(r"\s+", "", seq.upper())
    m = re.search(r"([ACDEFGHIKLMNPQRSTVWY]{80,})$", seq)
    return m.group(1) if m else None


def write_clean_fasta(src_fasta: Path, dst_fasta: Path) -> int:
    records: list[tuple[str, str]] = []
    header = None
    seq_parts: list[str] = []
    for line in src_fasta.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                raw = "".join(seq_parts)
                clean = clean_aa_sequence(raw)
                if clean:
                    records.append((header, clean))
            header = line[1:].strip()
            seq_parts = []
        elif header is not None:
            seq_parts.append(line.strip())
    if header is not None:
        raw = "".join(seq_parts)
        clean = clean_aa_sequence(raw)
        if clean:
            records.append((header, clean))

    dst_fasta.parent.mkdir(parents=True, exist_ok=True)
    with dst_fasta.open("w", encoding="utf-8") as fp:
        for h, s in records:
            fp.write(f">{h}\n{s}\n")
    return len(records)


def parse_plddt_scores(pdb_dir: Path) -> list[float]:
    scores = []
    if not pdb_dir.is_dir():
        return scores
    for pdb in sorted(pdb_dir.rglob("*.pdb")):
        m = re.search(r"_plddt_([0-9.]+)\.pdb$", pdb.name)
        if m:
            scores.append(float(m.group(1)))
    return scores


def summarize(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    import numpy as np

    arr = np.array(scores, dtype=float)
    return {
        "n": int(len(arr)),
        "mean_plddt": float(arr.mean()),
        "median_plddt": float(np.median(arr)),
        "min_plddt": float(arr.min()),
        "max_plddt": float(arr.max()),
        "frac_plddt_gt_70": float((arr > 70).mean()),
    }


def eval_one(
    ckpt: Path,
    output_dir: Path,
    num_seqs: int,
    seq_len: int,
    max_iter: int,
    gpu: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "plddt_summary.json"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text())
        if cached.get("n", 0) > 0:
            return cached

    gen_dir = output_dir / "sequence_generation" / f"length_{seq_len}"
    esm_dir = output_dir / "esmfold_pdb"
    env = {
        **dict(__import__("os").environ),
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
        "PROJECT_ROOT": str(REPO_ROOT),
        "HF_HUB_OFFLINE": "1",
    }

    gen_cmd = [
        PYTHON,
        str(REPO_ROOT / "generate_dplm2.py"),
        "--model_name",
        str(ckpt),
        "--num_seqs",
        str(num_seqs),
        "--seq_lens",
        str(seq_len),
        "--max_iter",
        str(max_iter),
        "--sampling_strategy",
        "gumbel_argmax",
        "--saveto",
        str(output_dir),
        "--task",
        "sequence_generation",
        "--save_pdb",
        "false",
        "--batch_size",
        "50",
    ]
    print(f"[gen] {' '.join(gen_cmd)}", flush=True)
    if not (gen_dir / "aatype.fasta").is_file():
        subprocess.run(gen_cmd, cwd=str(REPO_ROOT), env=env, check=True)
    else:
        print(f"[skip gen] {gen_dir / 'aatype.fasta'} exists", flush=True)

    src_fasta = gen_dir / "aatype.fasta"
    if not src_fasta.is_file():
        raise FileNotFoundError(f"Missing generated fasta: {src_fasta}")
    clean_dir = output_dir / "sequence_generation_clean" / f"length_{seq_len}"
    clean_fasta = clean_dir / "aatype.fasta"
    n_clean = write_clean_fasta(src_fasta, clean_fasta)
    if n_clean == 0:
        raise RuntimeError(f"No valid AA sequences in {src_fasta}")

    esm_dir.mkdir(parents=True, exist_ok=True)
    plddt_cmd = [
        PYTHON,
        str(REPO_ROOT / "analysis" / "cal_plddt_dir.py"),
        "-i",
        str(clean_dir),
        "-o",
        str(esm_dir),
        "--max-tokens-per-batch",
        "512",
    ]
    print(f"[plddt] {' '.join(plddt_cmd)}", flush=True)
    subprocess.run(plddt_cmd, cwd=str(REPO_ROOT), env=env, check=True)

    scores = parse_plddt_scores(esm_dir)
    row = {
        "checkpoint": str(ckpt),
        "step": step_from_ckpt(ckpt),
        "n_seqs": num_seqs,
        **summarize(scores),
    }
    summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-seqs", type=int, default=15)
    parser.add_argument("--seq-lens", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    ckpt_paths = sorted(
        [Path(p) for p in args.checkpoints],
        key=lambda p: step_from_ckpt(p),
    )
    rows = []
    for ckpt in ckpt_paths:
        step = step_from_ckpt(ckpt)
        out = args.output_dir / f"eval_step_{step}"
        print(f"\n=== pLDDT @ step {step} ===", flush=True)
        rows.append(
            eval_one(
                ckpt,
                out,
                num_seqs=args.num_seqs,
                seq_len=args.seq_lens,
                max_iter=args.max_iter,
                gpu=args.gpu,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "plddt_by_checkpoint.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["| Step | Mean pLDDT | Median | pLDDT>70 | N |", "|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: x.get("step", 0)):
        if r.get("n", 0) == 0:
            lines.append(f"| {r.get('step','?')} | — | — | — | 0 |")
        else:
            lines.append(
                f"| {r['step']} | {r['mean_plddt']:.1f} | {r['median_plddt']:.1f} | "
                f"{r['frac_plddt_gt_70']*100:.0f}% | {r['n']} |"
            )
    md_path = args.output_dir / "plddt_by_checkpoint.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {json_path}\n{md_path}")


if __name__ == "__main__":
    main()
