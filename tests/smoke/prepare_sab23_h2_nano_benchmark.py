#!/usr/bin/env python3
"""Prepare SAb-23-H2-Nano benchmark files for DPLM-2 folding / inverse folding.

Creates data-bin/sab23_h2_nano/:
  - aatype.fasta   (nanobody H-chain sequences, for folding)
  - pdb_h/         (H-chain-only native PDBs)
  - struct.fasta   (structure tokens from native PDBs, for inverse folding)

Usage:
  conda activate dplm
  export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
  python tests/smoke/prepare_sab23_h2_nano_benchmark.py \\
    --dataset-dir ~/SAb-23-H2-Nano \\
    --struct-tokenizer checkpoints/struct_tokenizer
"""

from __future__ import annotations

import argparse
import sys

from common import REPO_ROOT, resolve_struct_tokenizer
from sab23_h2_nano_lib import BENCHMARK_DIR, DEFAULT_DATASET_DIR, prepare_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SAb-23-H2-Nano benchmark for DPLM-2")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to extracted SAb-23-H2-Nano directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BENCHMARK_DIR),
        help="Output benchmark directory (default: data-bin/sab23_h2_nano)",
    )
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Only write aatype.fasta and pdb_h/ (no struct.fasta)",
    )
    args = parser.parse_args()

    from pathlib import Path

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    stok = None if args.skip_tokenize else resolve_struct_tokenizer(args.struct_tokenizer)
    paths = prepare_benchmark(
        dataset_dir=dataset_dir,
        benchmark_dir=Path(args.output_dir),
        struct_tokenizer_path=stok,
        skip_tokenize=args.skip_tokenize,
    )
    print("[PASS] SAb-23-H2-Nano benchmark prepared")
    print(f"  repo: {REPO_ROOT}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
