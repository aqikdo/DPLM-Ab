#!/usr/bin/env python3
"""Merge NGS VHH (sequence-only) and structural VHH HF datasets for mixed training.

Adds ``data_source`` (``ngs`` | ``struct``). NGS rows get null struct fields;
struct rows keep aa_seq/struct_seq.

Example:
  python scripts/prepare_vhh_ngs_struct_mixed.py \
    --ngs-dir data-bin/unpaired_ngs_vhh_mmseqs_clean_mfdesign_nano \
    --struct-dir data-bin/mfdesign_vhh_struct \
    --output-dir data-bin/vhh_ngs_mfdesign_mixed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_split(path: Path, split: str) -> Dataset:
    disk = path / split
    if not disk.is_dir():
        raise FileNotFoundError(f"Missing split: {disk}")
    return load_from_disk(str(disk))


def _prepare_ngs(ds: Dataset, limit: int | None) -> Dataset:
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    def map_row(row):
        seq = row.get("aa_seq") or row.get("seq")
        return {
            "aa_seq": seq,
            "struct_seq": None,
            "length": int(row["length"]),
            "data_source": "ngs",
            "pdb_name": None,
            "cluster": None,
        }

    return ds.map(map_row, remove_columns=ds.column_names, num_proc=1)


def _prepare_struct(ds: Dataset, limit: int | None) -> Dataset:
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    def map_row(row):
        return {
            "aa_seq": row["aa_seq"],
            "struct_seq": row["struct_seq"],
            "length": int(row["length"]),
            "data_source": "struct",
            "pdb_name": row.get("pdb_name"),
            "cluster": row.get("cluster"),
        }

    return ds.map(map_row, remove_columns=ds.column_names, num_proc=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NGS + struct mixed VHH dataset")
    parser.add_argument("--ngs-dir", type=str, required=True)
    parser.add_argument("--struct-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ngs-limit", type=int, default=None)
    parser.add_argument("--struct-limit", type=int, default=None)
    args = parser.parse_args()

    ngs_dir = Path(args.ngs_dir).expanduser()
    struct_dir = Path(args.struct_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "ngs_dir": str(ngs_dir.resolve()),
        "struct_dir": str(struct_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
    }

    for split in ("train", "valid"):
        ngs = _prepare_ngs(_load_split(ngs_dir, split), args.ngs_limit)
        struct = _prepare_struct(_load_split(struct_dir, split), args.struct_limit)
        # Align features before concat
        ngs = ngs.cast(struct.features)
        merged = concatenate_datasets([ngs, struct])
        merged.save_to_disk(str(out_dir / split), num_proc=1)
        n_ngs = sum(1 for x in merged["data_source"] if x == "ngs")
        n_struct = sum(1 for x in merged["data_source"] if x == "struct")
        manifest[f"n_{split}"] = len(merged)
        manifest[f"n_{split}_ngs"] = n_ngs
        manifest[f"n_{split}_struct"] = n_struct
        print(f"{split}: total={len(merged)} ngs={n_ngs} struct={n_struct}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir}/train, valid/")


if __name__ == "__main__":
    main()
