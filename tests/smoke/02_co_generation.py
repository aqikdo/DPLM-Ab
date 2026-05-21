#!/usr/bin/env python3
"""Smoke test: short unconditional sequence-structure co-generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import DEFAULT_OUTPUT, load_dplm2, require_cuda
from generate_dplm2 import initialize_generation, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="DPLM-2 co-generation smoke test")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "co_generation")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument(
        "--sampling-strategy", type=str, default="annealing@2.0:0.1"
    )
    args = parser.parse_args()

    require_cuda()
    model, _, _ = load_dplm2(args.model, args.struct_tokenizer)
    device = next(model.parameters()).device

    input_tokens = initialize_generation(
        task="co_generation",
        num_seqs=args.num_seqs,
        length=args.length,
        tokenizer=model.tokenizer,
        device=device,
        batch_size=args.num_seqs,
    )[0]

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            input_tokens=input_tokens,
            max_iter=args.max_iter,
            sampling_strategy=args.sampling_strategy,
        )

    save_dir = args.output
    save_results(
        outputs=outputs,
        task="co_generation",
        save_dir=str(save_dir),
        tokenizer=model.tokenizer,
        struct_tokenizer=model.struct_tokenizer,
        save_pdb=True,
    )

    aatype = save_dir / "aatype.fasta"
    pdb_dir = save_dir / "pdb"
    assert aatype.exists(), f"Missing {aatype}"
    assert pdb_dir.exists() and any(pdb_dir.glob("*.pdb")), f"Missing PDB in {pdb_dir}"

    print("[PASS] co_generation")
    print(f"  output: {save_dir}")
    print(f"  length={args.length} num_seqs={args.num_seqs} max_iter={args.max_iter}")


if __name__ == "__main__":
    main()
