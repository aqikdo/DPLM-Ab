#!/usr/bin/env python3
"""ESM-IF1 / nanoFOLD invfold on nano43: sample H in complex with Ag context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import esm
import torch
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from esm.inverse_folding.multichain_util import (
    _concatenate_coords,
    load_complex_coords,
)
from esm.inverse_folding.util import CoordBatchConverter


def sample_target_chain(model, coords_dict, target_chain_id, temperature=0.2, device="cpu"):
    target_chain_len = coords_dict[target_chain_id].shape[0]
    all_coords = _concatenate_coords(coords_dict, target_chain_id)
    padding_pattern = ["<pad>"] * all_coords.shape[0]
    for i in range(target_chain_len):
        padding_pattern[i] = "<mask>"

    L = len(all_coords)
    batch_converter = CoordBatchConverter(model.decoder.dictionary)
    batch_coords, confidence, _, _, padding_mask = batch_converter(
        [(all_coords, None, None)], device=device
    )
    mask_idx = model.decoder.dictionary.get_idx("<mask>")
    sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=torch.long, device=device)
    sampled_tokens[0, 0] = model.decoder.dictionary.get_idx("<cath>")
    for i, c in enumerate(padding_pattern):
        sampled_tokens[0, i + 1] = model.decoder.dictionary.get_idx(c)

    incremental_state = dict()
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)
    with torch.no_grad():
        for i in range(1, L + 1):
            if sampled_tokens[0, i] != mask_idx:
                continue
            logits, _ = model.decoder(
                sampled_tokens[:, :i],
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1) / temperature
            probs = F.softmax(logits, dim=-1)
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
    sampled_seq = sampled_tokens[0, 1 : 1 + target_chain_len]
    return "".join([model.decoder.dictionary.get_tok(int(a)) for a in sampled_seq])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None, help="nanoFOLD state dict")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--ab-only",
        action="store_true",
        help="Sample H from single-chain PDB (ignore antigen context).",
    )
    args = ap.parse_args()

    samples = json.loads(args.samples_json.read_text())
    if args.limit:
        samples = samples[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ESM-IF1 on {device} ...", flush=True)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if args.ckpt is not None:
        print(f"load state_dict {args.ckpt}", flush=True)
        sd = torch.load(str(args.ckpt), map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    model = model.eval().to(device)

    pred_records = []
    for s in samples:
        sid = s["sample_id"]
        h, ag = s["heavy_chain"], s["antigen_chain"]
        ab_only = args.ab_only or s.get("ab_only")
        print("RUN", sid, h, ag, "ab_only=" + str(ab_only), flush=True)
        if ab_only:
            want = [h]
        else:
            want = [h] + list(s.get("antigen_chains") or [])
            if len(want) == 1:
                from biotite.structure import get_chains as _get_chains
                from esm.inverse_folding.util import load_structure as _load_structure

                present = list(
                    _get_chains(_load_structure(str(s["pdb_path"]), chain=None))
                )
                if ag in present:
                    want.append(ag)
                else:
                    want.extend([c for c in str(ag) if c in present and c != h])
            want = list(dict.fromkeys(want))
        coords, _native = load_complex_coords(str(s["pdb_path"]), want)
        seq = sample_target_chain(
            model, coords, h, temperature=args.temperature, device=device
        )
        pred_records.append(SeqRecord(Seq(seq), id=sid, description=""))

    out_fa = args.out_dir / "predictions.fasta"
    SeqIO.write(pred_records, str(out_fa), "fasta")
    print(f"wrote {len(pred_records)} -> {out_fa}")


if __name__ == "__main__":
    main()
