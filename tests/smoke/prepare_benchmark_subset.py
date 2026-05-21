#!/usr/bin/env python3
"""Extract a small CAMEO subset for fast benchmark smoke tests.

Requires full benchmark data from:
  bash scripts/download_metadata.sh

Usage:
  python tests/smoke/prepare_benchmark_subset.py --max-proteins 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio import SeqIO

from common import BENCHMARK_ROOT, CAMEO_DIR, CAMEO_MINI_DIR, REPO_ROOT


def subset_fasta(src: Path, dst: Path, max_proteins: int) -> int:
    records = list(SeqIO.parse(str(src), "fasta"))[:max_proteins]
    if not records:
        raise ValueError(f"No records in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(dst), "fasta")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=CAMEO_DIR,
        help="Full CAMEO dir (default: data-bin/cameo2022)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CAMEO_MINI_DIR,
        help="Mini subset output (default: data-bin/cameo2022_mini)",
    )
    parser.add_argument("--max-proteins", type=int, default=3)
    args = parser.parse_args()

    aatype_src = args.source_dir / "aatype.fasta"
    struct_src = args.source_dir / "struct.fasta"
    if not aatype_src.exists() or not struct_src.exists():
        raise SystemExit(
            f"Benchmark data missing under {args.source_dir}.\n"
            f"Run from repo root: bash scripts/download_metadata.sh"
        )

    n_aa = subset_fasta(aatype_src, args.output_dir / "aatype.fasta", args.max_proteins)
    n_st = subset_fasta(struct_src, args.output_dir / "struct.fasta", args.max_proteins)
    print(f"Wrote {n_aa} proteins to {args.output_dir} (aatype + struct)")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Benchmark root: {BENCHMARK_ROOT}")


if __name__ == "__main__":
    main()
