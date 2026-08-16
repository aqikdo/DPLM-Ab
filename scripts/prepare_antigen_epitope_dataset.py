#!/usr/bin/env python3
"""Attach antigen tokens and residue-level epitope labels to complex metadata.

Expected input CSV columns:
  - pdb_path
  - chain_id or antibody_chain
  - antigen_chain
  - aa_seq
Optional:
  - pdb_name

Output CSV adds:
  - antigen_aa_seq
  - antigen_struct_seq
  - epitope_mask
  - antigen_chain_map
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from Bio.PDB import PDBParser

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from byprot.models.utils import get_struct_tokenizer  # noqa: E402
from data.sabdab_nanobody import tokenize_chain_to_row  # noqa: E402

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def iter_polymer_residues(chain):
    for residue in chain:
        if residue.id[0] == " ":
            yield residue


def residue_to_aa(residue) -> str:
    return AA3_TO_1.get(residue.resname.upper(), "X")


def min_residue_distance(res_a, res_b) -> float:
    best = float("inf")
    for atom_a in res_a.get_atoms():
        for atom_b in res_b.get_atoms():
            dist = atom_a - atom_b
            if dist < best:
                best = dist
    return best


def derive_epitope_mask(
    structure_path: Path,
    antibody_chains: set[str],
    antigen_chains: list[str],
    threshold: float,
):
    structure = PDBParser(QUIET=True).get_structure(structure_path.stem, str(structure_path))
    model = structure[0]
    antibody_residues = []
    antigen_residues = []
    antigen_seq = []
    chain_map = []

    for chain in model:
        cid = chain.id.strip()
        if cid in antibody_chains:
            antibody_residues.extend(list(iter_polymer_residues(chain)))
        if cid in antigen_chains:
            residues = list(iter_polymer_residues(chain))
            antigen_residues.extend(residues)
            antigen_seq.extend(residue_to_aa(r) for r in residues)
            chain_map.extend([cid] * len(residues))

    mask = []
    for ag_res in antigen_residues:
        is_epi = any(
            min_residue_distance(ag_res, ab_res) <= threshold
            for ab_res in antibody_residues
        )
        mask.append(int(is_epi))
    return "".join(antigen_seq), ",".join(str(x) for x in mask), ",".join(chain_map)


def tokenize_antigen_chains(
    structure_path: Path,
    antigen_chains: list[str],
    struct_tokenizer,
    work_dir: Path,
):
    aa_parts = []
    struct_parts = []
    for chain_id in antigen_chains:
        tok = tokenize_chain_to_row(
            structure_path,
            chain_id,
            struct_tokenizer,
            pdb_name=f"{structure_path.stem}_{chain_id}",
            work_dir=work_dir,
        )
        aa_parts.append(tok["aa_seq"])
        struct_parts.append(tok["struct_seq"])
    return "".join(aa_parts), ",".join(
        part for part in struct_parts if part
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--struct-tokenizer", type=Path, default=REPO_ROOT / "checkpoints" / "struct_tokenizer")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    struct_tokenizer = get_struct_tokenizer(str(args.struct_tokenizer))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        struct_tokenizer = struct_tokenizer.cuda().eval()
    else:
        struct_tokenizer = struct_tokenizer.cpu().eval()
    work_dir = args.output_csv.parent / "_antigen_chain_pdbs"
    rows = []
    for _, row in df.iterrows():
        pdb_path = Path(row["pdb_path"])
        ab_chain = str(row.get("chain_id") or row.get("antibody_chain") or "").strip()
        ag_chains = [
            c.strip()
            for c in str(row.get("antigen_chain") or "").replace("|", ",").split(",")
            if c.strip()
        ]
        if not pdb_path.is_file() or not ab_chain or not ag_chains:
            rows.append(
                {
                    **row.to_dict(),
                    "antigen_aa_seq": None,
                    "antigen_struct_seq": None,
                    "epitope_mask": None,
                    "antigen_chain_map": None,
                }
            )
            continue
        antigen_seq, epitope_mask, chain_map = derive_epitope_mask(
            pdb_path,
            antibody_chains={ab_chain},
            antigen_chains=ag_chains,
            threshold=args.threshold,
        )
        tokenized_ag_seq, tokenized_ag_struct = tokenize_antigen_chains(
            pdb_path, ag_chains, struct_tokenizer, work_dir
        )
        rows.append(
            {
                **row.to_dict(),
                "antigen_aa_seq": tokenized_ag_seq or antigen_seq,
                "antigen_struct_seq": tokenized_ag_struct,
                "epitope_mask": epitope_mask,
                "antigen_chain_map": chain_map,
            }
        )

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} rows={len(out)}")


if __name__ == "__main__":
    main()
