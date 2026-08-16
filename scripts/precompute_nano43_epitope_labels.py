#!/usr/bin/env python3
"""Precompute 5Å GT epitope masks for MFDesign nano43 eval complexes.

Writes data-bin/benchmarks/mfdesign_nano_vhh/epitope_labels.json keyed by sample_id.
Uses the same derive_epitope_mask logic as prepare_antigen_epitope_dataset.py.
When BioPython polymer length/seq differs from the cached antigen_aa_seq
(e.g. nonstandard residues), project epitope labels onto the cache sequence.
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from prepare_antigen_epitope_dataset import derive_epitope_mask  # noqa: E402

BENCH = REPO / "data-bin" / "benchmarks" / "mfdesign_nano_vhh"
THRESHOLD = 5.0


def _antigen_chains(raw) -> list[str]:
    """Match prepare_nano_antigen: list(antigen_chain) for string ids."""
    if isinstance(raw, (list, tuple)):
        return [str(c) for c in raw]
    return list(str(raw))


def project_epitope(src_seq: str, src_epi: list[int], tgt_seq: str) -> list[int]:
    """Map epitope labels from derive_seq onto cached antigen_aa_seq."""
    if src_seq == tgt_seq and len(src_epi) == len(tgt_seq):
        return list(src_epi)
    out = [0] * len(tgt_seq)
    sm = SequenceMatcher(None, src_seq, tgt_seq, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for di in range(i2 - i1):
                out[j1 + di] = int(src_epi[i1 + di])
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for di in range(n):
                out[j1 + di] = int(src_epi[i1 + di])
        # delete/insert: leave zeros on tgt
    return out


def main():
    rows_path = BENCH / "antigen_eval_rows.json"
    if not rows_path.is_file():
        raise FileNotFoundError(f"missing {rows_path}; run eval prepare first")
    rows = json.loads(rows_path.read_text())
    out = {}
    for r in rows:
        sid = r["sample_id"]
        pdb_path = Path(r["pdb_path"])
        ab_chain = str(r["chain_id"]).strip()
        ag_chains = _antigen_chains(r["antigen_chain"])

        aa_seq, epi_csv, _chain_map = derive_epitope_mask(
            pdb_path,
            antibody_chains={ab_chain},
            antigen_chains=ag_chains,
            threshold=THRESHOLD,
        )
        epi = [int(x) for x in epi_csv.split(",") if len(x) > 0]
        expected = r["antigen_aa_seq"]
        if len(epi) != len(aa_seq):
            raise RuntimeError(
                f"{sid}: derive epi len {len(epi)} != derive aa {len(aa_seq)}"
            )
        if aa_seq != expected or len(epi) != len(expected):
            print(
                f"[warn] {sid}: projecting epitope "
                f"derive_len={len(aa_seq)} -> cache_len={len(expected)}",
                flush=True,
            )
            epi = project_epitope(aa_seq, epi, expected)
        out[sid] = {
            "entry_id": r["entry_id"],
            "pdb_path": str(pdb_path),
            "antibody_chain": ab_chain,
            "antigen_chains": ag_chains,
            "n_epitope": int(sum(epi)),
            "ag_len": len(epi),
            "epitope_mask": epi,
            "threshold": THRESHOLD,
        }
        print(f"{sid}: n_epi={sum(epi)}/{len(epi)}", flush=True)

    out_path = BENCH / "epitope_labels.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} n={len(out)}", flush=True)


if __name__ == "__main__":
    main()
