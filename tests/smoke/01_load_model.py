#!/usr/bin/env python3
"""Smoke test: load DPLM-2 weights and struct tokenizer."""

from __future__ import annotations

import argparse

from common import DEFAULT_MODEL, load_dplm2, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Load DPLM-2 checkpoint smoke test")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--struct-tokenizer", type=str, default=None)
    args = parser.parse_args()

    require_cuda()
    model, model_dir, stok_dir = load_dplm2(args.model, args.struct_tokenizer)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("[PASS] load_model")
    print(f"  model: {model_dir}")
    print(f"  struct_tokenizer (config): {stok_dir}")
    print(f"  parameters: {n_params:.2f}M")
    print(f"  device: {next(model.parameters()).device}")

    try:
        _ = model.struct_tokenizer
        print("  struct_tokenizer load: OK")
    except ModuleNotFoundError as e:
        print(f"  struct_tokenizer load: SKIP ({e})")
        print("  Install: pip install fairscale  (required for co-generation / PDB export)")


if __name__ == "__main__":
    main()
