#!/usr/bin/env python3
"""Smoke test: forward folding on CAMEO mini benchmark subset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from peft.peft_model import PeftModel
from tqdm import tqdm

from common import (
    CAMEO_MINI_DIR,
    DEFAULT_OUTPUT,
    benchmark_available,
    load_dplm2,
    require_cuda,
)
from generate_dplm2 import initialize_conditional_generation, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="DPLM-2 folding benchmark smoke test")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=CAMEO_MINI_DIR / "aatype.fasta",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "folding")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--run-eval", action="store_true", help="Run evaluator (needs metadata)")
    args = parser.parse_args()

    if not args.input_fasta.exists():
        print(
            "[SKIP] folding_benchmark: benchmark FASTA not found.\n"
            f"  Expected: {args.input_fasta}\n"
            "  Steps:\n"
            "    1) bash scripts/download_metadata.sh\n"
            "    2) python tests/smoke/prepare_benchmark_subset.py --max-proteins 3",
            file=sys.stderr,
        )
        sys.exit(0)

    require_cuda()
    model, _, _ = load_dplm2(args.model, args.struct_tokenizer)
    if isinstance(model.net, PeftModel):
        model.net = model.net.merge_and_unload()
    tokenizer = model.tokenizer
    device = next(model.parameters()).device

    class Args:
        task = "folding"
        batch_size = args.batch_size

    batches, name_lists = initialize_conditional_generation(
        str(args.input_fasta), tokenizer, device, args=Args(), model=model
    )

    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(batches, desc="folding")):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=args.max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=batch["partial_mask"],
            )
        save_results(
            outputs=outputs,
            task="folding",
            save_dir=str(save_dir),
            headers=name_lists[i],
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=True,
            continue_write=i > 0,
        )

    aatype = save_dir / "aatype.fasta"
    pdb_dir = save_dir / "pdb"
    assert aatype.exists() and pdb_dir.exists()

    print("[PASS] folding_benchmark")
    print(f"  input: {args.input_fasta}")
    print(f"  output: {save_dir}")

    if args.run_eval:
        from common import benchmark_metadata_available

        if not benchmark_metadata_available():
            print("[WARN] --run-eval skipped: data-bin/metadata/pdb_afdb_cameo.csv missing")
            return
        import subprocess

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "src/byprot/utils/protein/evaluator_dplm2.py"),
            "-cn",
            "forward_folding",
            f"inference.input_fasta_dir={save_dir}",
            "inference.struct_tokenizer.exp_path="
            + str(model.cfg.struct_tokenizer.exp_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), check=True)


if __name__ == "__main__":
    main()
