#!/usr/bin/env python3
"""Benchmark DPLM-2 folding and inverse folding on single-chain nanobodies.

Benchmarks:
  - igfold (default): IgFold paper nanobody set, ~/benchmark/nano/IgFold (71 targets)
  - sab23: SAb-23-H2-Nano (27 targets)

Usage:
  conda activate dplm
  export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH

  # IgFold nano (default)
  python tests/smoke/07_sab23_h2_nano_benchmark.py \\
    --benchmark igfold \\
    --benchmark-root ~/benchmark \\
    --model checkpoints/dplm2_650m \\
    --struct-tokenizer checkpoints/struct_tokenizer

  # SAb-23-H2-Nano
  python tests/smoke/07_sab23_h2_nano_benchmark.py --benchmark sab23 \\
    --dataset-dir ~/SAb-23-H2-Nano

  # Metrics only
  python tests/smoke/07_sab23_h2_nano_benchmark.py --metrics-only \\
    --benchmark igfold --output tests/smoke_outputs/igfold_nano
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from Bio import SeqIO
from peft.peft_model import PeftModel
from tqdm import tqdm

from common import DEFAULT_OUTPUT, load_dplm2, require_cuda
from generate_dplm2 import initialize_conditional_generation, save_results
from sab23_h2_nano_lib import (
    BENCHMARK_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_IGFOLD_BENCHMARK_ROOT,
    DEFAULT_IGFOLD_XTAL_DIR,
    IGFOLD_BENCHMARK_DIR,
    compute_folding_metrics,
    compute_inverse_folding_metrics,
    load_cdr_numbering,
    load_cdr_regions,
    load_native_sequences,
    prepare_benchmark,
    prepare_igfold_nano_benchmark,
    print_region_table,
    save_cdr_regions,
    summarize_metrics,
    summarize_region_metrics,
)


def _run_task(
    model,
    tokenizer,
    task: str,
    input_fasta: Path,
    save_dir: Path,
    max_iter: int,
    batch_size: int,
    save_pdb: bool,
) -> None:
    device = next(model.parameters()).device

    class Args:
        pass

    args_ns = Args()
    args_ns.task = task
    args_ns.batch_size = batch_size

    batches, name_lists = initialize_conditional_generation(
        str(input_fasta), tokenizer, device, args=args_ns, model=model
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(tqdm(batches, desc=task)):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=batch["partial_mask"],
            )
        save_results(
            outputs=outputs,
            task=task,
            save_dir=str(save_dir),
            headers=name_lists[i],
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=save_pdb,
            continue_write=i > 0,
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPLM-2 folding & inverse folding on nanobody benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        choices=("igfold", "sab23"),
        default="igfold",
        help="Benchmark suite (default: igfold = ~/benchmark/nano/IgFold)",
    )
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=str(DEFAULT_IGFOLD_BENCHMARK_ROOT),
        help="IgFold benchmark root (nano/IgFold/*.pdb)",
    )
    parser.add_argument(
        "--xtal-dir",
        type=str,
        default=str(DEFAULT_IGFOLD_XTAL_DIR),
        help="IgFold xtal/July2021_nano natives and fastas",
    )
    parser.add_argument(
        "--native-dir",
        type=str,
        default=None,
        help="Optional directory with extra native PDBs for IgFold benchmark",
    )
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit samples (-1 = all)")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-folding", action="store_true")
    parser.add_argument("--skip-inverse-folding", action="store_true")
    parser.add_argument(
        "--run-inverse-folding",
        action="store_true",
        help="Also run inverse folding (default off for --benchmark igfold)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics from existing outputs",
    )
    args = parser.parse_args()

    is_igfold = args.benchmark == "igfold"
    benchmark_label = "IgFold nanobody" if is_igfold else "SAb-23-H2-Nano"
    benchmark_dir = Path(
        args.benchmark_dir
        or (IGFOLD_BENCHMARK_DIR if is_igfold else BENCHMARK_DIR)
    )
    output_root = Path(
        args.output
        or (DEFAULT_OUTPUT / ("igfold_nano" if is_igfold else "sab23_h2_nano"))
    )
    dataset_dir = Path(args.dataset_dir)
    benchmark_root = Path(args.benchmark_root)
    xtal_dir = Path(args.xtal_dir)
    native_extra_dir = Path(args.native_dir) if args.native_dir else None
    skip_invfold = args.skip_inverse_folding or (is_igfold and not args.run_inverse_folding)
    folding_dir = output_root / "folding"
    invfold_dir = output_root / "inverse_folding"

    aatype_fasta = benchmark_dir / "aatype.fasta"
    struct_fasta = benchmark_dir / "struct.fasta"
    pdb_h_dir = benchmark_dir / "pdb_h"

    if not args.metrics_only:
        if not args.skip_prepare:
            stok = None
            need_struct_tok = (not is_igfold) or args.run_inverse_folding
            if need_struct_tok:
                try:
                    from common import resolve_struct_tokenizer

                    stok = resolve_struct_tokenizer(args.struct_tokenizer)
                except FileNotFoundError as e:
                    print(f"[ERROR] {e}", file=sys.stderr)
                    sys.exit(1)
            if is_igfold:
                prepare_igfold_nano_benchmark(
                    benchmark_root=benchmark_root,
                    benchmark_dir=benchmark_dir,
                    xtal_dir=xtal_dir,
                    native_extra_dir=native_extra_dir,
                    struct_tokenizer_path=stok,
                    skip_tokenize=not args.run_inverse_folding,
                )
            else:
                prepare_benchmark(
                    dataset_dir=dataset_dir,
                    benchmark_dir=benchmark_dir,
                    struct_tokenizer_path=stok,
                    skip_tokenize=False,
                )
        elif not aatype_fasta.exists():
            prep_cmd = (
                "python tests/smoke/prepare_igfold_nano_benchmark.py "
                f"--benchmark-root {benchmark_root}"
                if is_igfold
                else "python tests/smoke/prepare_sab23_h2_nano_benchmark.py "
                f"--dataset-dir {dataset_dir}"
            )
            print(
                "[ERROR] Benchmark data missing. Run without --skip-prepare first:\n"
                f"  {prep_cmd}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not skip_invfold and not struct_fasta.exists():
            print(
                f"[WARN] {struct_fasta} missing; inverse folding will be skipped. "
                "Re-run prepare with --run-inverse-folding and --struct-tokenizer.",
                file=sys.stderr,
            )

        if args.max_samples > 0:
            records = []
            for rec in SeqIO.parse(str(aatype_fasta), "fasta"):
                records.append(rec)
                if len(records) >= args.max_samples:
                    break
            mini = benchmark_dir / "aatype_mini.fasta"
            mini_struct = benchmark_dir / "struct_mini.fasta"
            SeqIO.write(records, str(mini), "fasta")
            struct_recs = {r.id: r for r in SeqIO.parse(str(struct_fasta), "fasta")}
            mini_struct_recs = [struct_recs[r.id] for r in records if r.id in struct_recs]
            SeqIO.write(mini_struct_recs, str(mini_struct), "fasta")
            aatype_fasta = mini
            struct_fasta = mini_struct

        require_cuda()
        model, _, _ = load_dplm2(args.model, args.struct_tokenizer)
        if isinstance(model.net, PeftModel):
            model.net = model.net.merge_and_unload()
        tokenizer = model.tokenizer

        if not args.skip_folding:
            _run_task(
                model,
                tokenizer,
                task="folding",
                input_fasta=aatype_fasta,
                save_dir=folding_dir,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                save_pdb=True,
            )

        if not skip_invfold and struct_fasta.exists():
            _run_task(
                model,
                tokenizer,
                task="inverse_folding",
                input_fasta=struct_fasta,
                save_dir=invfold_dir,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                save_pdb=False,
            )
        elif not skip_invfold:
            print("[WARN] struct.fasta missing; skip inverse folding", file=sys.stderr)

    has_native_pdb = pdb_h_dir.is_dir() and any(pdb_h_dir.glob("*.pdb"))
    if not has_native_pdb:
        print(
            f"[WARN] No native PDBs in {pdb_h_dir}; folding/inverse metrics vs crystal skipped.",
            file=sys.stderr,
        )

    native_seqs = load_native_sequences(benchmark_dir)
    summary = {"benchmark_dir": str(benchmark_dir), "output": str(output_root)}

    cdr_regions_path = benchmark_dir / "cdr_regions.json"
    if not cdr_regions_path.exists():
        if not is_igfold and dataset_dir.exists():
            save_cdr_regions(dataset_dir, benchmark_dir)
        else:
            print(
                f"[WARN] {cdr_regions_path} missing; regional metrics skipped.\n"
                "  Re-run prepare for this benchmark.",
                file=sys.stderr,
            )
    cdr_regions = load_cdr_regions(benchmark_dir) if cdr_regions_path.exists() else None
    cdr_numbering = load_cdr_numbering(benchmark_dir) if cdr_regions_path.exists() else "sequence_0based"

    folding_pred = folding_dir / "pdb"
    if folding_pred.exists() and has_native_pdb:
        fold_rows = compute_folding_metrics(
            folding_pred, pdb_h_dir, cdr_regions=cdr_regions, cdr_numbering=cdr_numbering
        )
        _write_csv(output_root / "folding_metrics.csv", fold_rows)
        tm_summary = summarize_metrics(fold_rows, "bb_tmscore")
        rmsd_summary = summarize_metrics(fold_rows, "bb_rmsd")
        fold_summary = {
            "bb_tmscore": tm_summary,
            "bb_rmsd": rmsd_summary,
            "tmscore_ge_0.8": sum(
                1 for r in fold_rows if (r.get("bb_tmscore") or 0) >= 0.8
            ),
        }
        if cdr_regions:
            fold_summary["regions_ca_rmsd"] = summarize_region_metrics(
                fold_rows, "ca_rmsd"
            )
        summary["folding"] = fold_summary
        print(f"\n[FOLDING] {benchmark_label}")
        n = tm_summary.get("count", 0)
        n_pred = len(list(folding_pred.glob("*.pdb")))
        print(f"  predictions: {n_pred}  |  with native (scored): {n}")
        if n > 0:
            print(f"  mean bb_tmscore:   {tm_summary['mean']:.3f}")
            print(f"  median bb_tmscore: {tm_summary['median']:.3f}")
            print(f"  mean bb_rmsd:      {rmsd_summary['mean']:.3f} A")
            print(f"  median bb_rmsd:    {rmsd_summary['median']:.3f} A")
            print(f"  TM>=0.8:           {fold_summary['tmscore_ge_0.8']}/{n}")
        if cdr_regions and fold_summary.get("regions_ca_rmsd"):
            print_region_table(
                "[FOLDING] Regional CA RMSD (Å, align on Fr)",
                fold_summary["regions_ca_rmsd"],
            )
            cdr_note = (
                "Chothia CDR1/2/3; CDR3 length from IgFold stats when available"
                if is_igfold
                else "CDR1/2/3 from design FASTA X masks"
            )
            print(f"  (Fr=framework; {cdr_note}; RMSD after Fr alignment)")
    elif folding_pred.exists():
        print("[WARN] Folding predictions exist but no native PDBs; skip folding metrics")
    else:
        print("[WARN] Folding predictions not found; skip folding metrics")

    inv_aatype = invfold_dir / "aatype.fasta"
    if inv_aatype.exists():
        inv_rows = compute_inverse_folding_metrics(
            inv_aatype, native_seqs, cdr_regions=cdr_regions
        )
        _write_csv(output_root / "inverse_folding_metrics.csv", inv_rows)
        inv_summary = summarize_metrics(inv_rows, "seq_recovery")
        inv_summary["recovery_ge_0.5"] = sum(
            1 for r in inv_rows if (r.get("seq_recovery") or 0) >= 0.5
        )
        if cdr_regions:
            inv_summary["regions_seq_recovery"] = summarize_region_metrics(
                inv_rows, "seq_recovery"
            )
        summary["inverse_folding"] = inv_summary
        print(f"\n[INVERSE FOLDING] {benchmark_label}")
        print(f"  samples: {inv_summary.get('count', 0)}")
        if inv_summary.get("count", 0) > 0:
            print(f"  mean seq_recovery:   {inv_summary['mean']:.3f}")
            print(f"  median seq_recovery: {inv_summary['median']:.3f}")
            print(f"  recovery>=0.5:       {inv_summary['recovery_ge_0.5']}/{inv_summary['count']}")
        if cdr_regions and inv_summary.get("regions_seq_recovery"):
            print_region_table(
                "[INVERSE FOLDING] Regional sequence recovery",
                inv_summary["regions_seq_recovery"],
            )
    else:
        print("[WARN] Inverse folding predictions not found; skip invfold metrics")

    summary_path = output_root / "summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[PASS] Results written to {output_root}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
