#!/usr/bin/env python3
"""Prepare strict SAbDab nanobody (VHH) data for DPLM-2 LoRA training.

Writes ``data-bin/sabdab_nanobody/`` with metadata.csv, train/, valid/, rejected.jsonl.

Example:
  conda activate dplm
  export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
  python scripts/prepare_sabdab_nanobody_for_dplm2.py \\
    --sabdab-dir ~/sabdab \\
    --output-dir data-bin/sabdab_nanobody \\
    --struct-tokenizer checkpoints/struct_tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from data.sabdab_nanobody import (  # noqa: E402
    assign_train_valid_split,
    build_metadata_and_filter,
    load_exclude_ids,
    rejection_report,
    tokenize_rows,
    write_rejected_jsonl,
)
from byprot.models.utils import get_struct_tokenizer  # noqa: E402


def _resolve_struct_tokenizer(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser()
        if p.is_dir():
            return p.resolve()
    for cand in (
        REPO_ROOT / "checkpoints" / "struct_tokenizer",
        REPO_ROOT / "checkpoints",
    ):
        if cand.is_dir() and ((cand / "config.json").exists() or cand.name == "struct_tokenizer"):
            return cand.resolve()
    raise FileNotFoundError(
        "struct_tokenizer not found; pass --struct-tokenizer or place under checkpoints/"
    )


def save_hf_splits(rows: list[dict], output_dir: Path) -> dict[str, int]:
    train_rows = [r for r in rows if r.get("split") == "train"]
    valid_rows = [r for r in rows if r.get("split") == "valid"]
    cols = ["pdb_name", "aa_seq", "struct_seq", "length", "cluster", "split"]
    for split_name, split_rows in ("train", train_rows), ("valid", valid_rows):
        if not split_rows:
            continue
        df = pd.DataFrame(split_rows)
        keep = [c for c in cols if c in df.columns]
        ds = Dataset.from_pandas(df[keep], preserve_index=False)
        out = output_dir / split_name
        ds.save_to_disk(str(out), num_proc=1)
    return {"n_train": len(train_rows), "n_valid": len(valid_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SAbDab nanobody HF dataset for DPLM-2")
    parser.add_argument("--sabdab-dir", type=str, default=str(Path.home() / "sabdab"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "data-bin" / "sabdab_nanobody"),
    )
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--min-len", type=int, default=80)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Max entries to scan (smoke)")
    parser.add_argument(
        "--tokenize-limit",
        type=int,
        default=None,
        help="Max structures to tokenize after filtering (smoke)",
    )
    parser.add_argument(
        "--exclude-ids-file",
        type=str,
        default=None,
        help="Optional pdb ids to exclude (txt/json/jsonl)",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only filter and write metadata; skip struct tokenization",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume tokenization using existing metadata.csv struct_seq",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sabdab_dir = Path(args.sabdab_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sabdab_dir.is_dir():
        print(f"[ERROR] SAbDab dir not found: {sabdab_dir}", file=sys.stderr)
        sys.exit(1)

    exclude_ids = load_exclude_ids(
        Path(args.exclude_ids_file) if args.exclude_ids_file else None
    )
    if exclude_ids:
        print(f"Excluding {len(exclude_ids)} pdb ids from {args.exclude_ids_file}")

    accepted, rejected = build_metadata_and_filter(
        sabdab_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        exclude_ids=exclude_ids,
        limit=args.limit,
    )
    print(f"Filtered: included={len(accepted)} rejected={len(rejected)}")
    print(rejection_report(rejected))

    assign_train_valid_split(accepted, val_ratio=args.val_ratio, seed=args.seed)

    metadata_path = output_dir / "metadata.csv"
    rejected_path = output_dir / "rejected.jsonl"

    if args.resume and metadata_path.is_file():
        prev = pd.read_csv(metadata_path)
        done = {
            r["pdb_name"]: r
            for _, r in prev.iterrows()
            if pd.notna(r.get("struct_seq")) and str(r.get("struct_seq", "")).strip()
        }
        for i, row in enumerate(accepted):
            if row["pdb_name"] in done:
                accepted[i] = {**row, **done[row["pdb_name"]].to_dict()}
        print(f"Resume: {len(done)} rows with struct_seq from {metadata_path}")

    tokenize_failed: list[dict] = []
    if not args.filter_only:
        stok_path = _resolve_struct_tokenizer(args.struct_tokenizer)
        print(f"Loading struct tokenizer from {stok_path}")
        struct_tokenizer = get_struct_tokenizer(str(stok_path))
        if args.device.startswith("cuda") and torch.cuda.is_available():
            struct_tokenizer = struct_tokenizer.cuda().eval()
        else:
            struct_tokenizer = struct_tokenizer.cpu().eval()

        need_tok = [r for r in accepted if not r.get("struct_seq")]
        if args.tokenize_limit is not None:
            need_tok = need_tok[: args.tokenize_limit]
        print(f"Tokenizing {len(need_tok)} structures...")
        work_dir = output_dir / "_chain_pdbs"
        tok_ok, tokenize_failed = tokenize_rows(
            need_tok,
            struct_tokenizer,
            work_dir=work_dir,
        )
        # merge tokenized back into accepted
        tok_by_name = {r["pdb_name"]: r for r in tok_ok}
        if args.tokenize_limit is not None:
            accepted = [tok_by_name[r["pdb_name"]] for r in need_tok if r["pdb_name"] in tok_by_name]
        else:
            accepted = [tok_by_name.get(r["pdb_name"], r) for r in accepted]
            accepted = [r for r in accepted if r.get("struct_seq")]
        rejected.extend(tokenize_failed)

    write_rejected_jsonl(rejected_path, rejected)

    df_meta = pd.DataFrame(accepted)
    df_meta.to_csv(metadata_path, index=False)

    counts = save_hf_splits(accepted, output_dir)
    manifest = {
        "sabdab_dir": str(sabdab_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_scanned": len(accepted) + len([r for r in rejected if r.get("reason") != "tokenize_error"]),
        "n_included": len(accepted),
        "n_rejected": len(rejected),
        **counts,
        "val_ratio": args.val_ratio,
        "min_len": args.min_len,
        "max_len": args.max_len,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {metadata_path}, {output_dir}/train, {output_dir}/valid")


if __name__ == "__main__":
    main()
