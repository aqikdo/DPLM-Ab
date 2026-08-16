#!/usr/bin/env python3
"""Prepare IgFold nanobody benchmark (benchmark/nano/IgFold, 71 targets) for DPLM-2.

Creates data-bin/igfold_nano/:
  - aatype.fasta   (sequences for all 71; from xtal fasta or IgFold PDB)
  - pdb_h/         (native crystal structures when available)
  - struct.fasta   (structure tokens from native PDBs, for inverse folding)
  - cdr_regions.json (Chothia CDR1/2/3; CDR3 length from IgFold stats.csv when present)

Usage:
  conda activate dplm
  export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
  python tests/smoke/prepare_igfold_nano_benchmark.py \\
    --benchmark-root ~/benchmark \\
    --struct-tokenizer checkpoints/struct_tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import REPO_ROOT, resolve_struct_tokenizer
from sab23_h2_nano_lib import (
    DEFAULT_IGFOLD_BENCHMARK_ROOT,
    DEFAULT_IGFOLD_XTAL_DIR,
    IGFOLD_BENCHMARK_DIR,
    prepare_igfold_nano_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IgFold nanobody benchmark for DPLM-2")
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=str(DEFAULT_IGFOLD_BENCHMARK_ROOT),
        help="Root of Graylab benchmark tarball (contains nano/IgFold/*.pdb)",
    )
    parser.add_argument(
        "--xtal-dir",
        type=str,
        default=str(DEFAULT_IGFOLD_XTAL_DIR),
        help="IgFold xtal/July2021_nano (native PDB + FASTA subset)",
    )
    parser.add_argument(
        "--native-dir",
        type=str,
        default=None,
        help="Optional extra directory with native {id}.pdb or {id}_trunc.pdb",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(IGFOLD_BENCHMARK_DIR),
        help="Prepared benchmark output (default: data-bin/igfold_nano)",
    )
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Also build struct.fasta (only needed for inverse folding)",
    )
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    if not (benchmark_root / "nano" / "IgFold").is_dir():
        print(f"[ERROR] Missing {benchmark_root / 'nano' / 'IgFold'}", file=sys.stderr)
        sys.exit(1)

    skip_tokenize = not args.tokenize
    stok = None if skip_tokenize else resolve_struct_tokenizer(args.struct_tokenizer)
    native_extra = Path(args.native_dir) if args.native_dir else None
    paths = prepare_igfold_nano_benchmark(
        benchmark_root=benchmark_root,
        benchmark_dir=Path(args.output_dir),
        xtal_dir=Path(args.xtal_dir),
        native_extra_dir=native_extra,
        struct_tokenizer_path=stok,
        skip_tokenize=skip_tokenize,
    )
    manifest = __import__("json").loads(paths["manifest"].read_text(encoding="utf-8"))
    n = manifest["n_samples"]
    n_native = manifest["n_with_native"]
    print("[PASS] IgFold nanobody benchmark prepared")
    print(f"  repo: {REPO_ROOT}")
    print(f"  targets: {n} (native PDB for metrics: {n_native})")
    if n_native < n:
        print(
            f"  [NOTE] {n - n_native} targets lack crystal PDB in xtal/native-dir; "
            "folding can still run, but metrics need natives in pdb_h/."
        )
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
