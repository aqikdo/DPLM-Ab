#!/usr/bin/env python3
"""Evaluate antigen-conditioned Ab co-generation on MFDesign nano43.

Task: given antigen aa+struct tokens, generate antibody aa+struct (joint),
report sequence recovery AAR / Fr / CDR1-3.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from Bio import SeqIO
from peft.peft_model import PeftModel
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "smoke"))
sys.path.insert(0, str(REPO / "scripts"))

from byprot.models.utils import get_struct_tokenizer  # noqa: E402
from common import require_cuda  # noqa: E402
from data.sabdab_nanobody import tokenize_chain_to_row  # noqa: E402
from eval_vhh_cdr_rmsd import to_lib_regions  # noqa: E402
from generate_dplm2 import save_results  # noqa: E402
from sab23_h2_nano_lib import (  # noqa: E402
    compute_folding_metrics,
    compute_inverse_folding_metrics,
    load_native_sequences,
    print_region_table,
    summarize_metrics,
    summarize_region_metrics,
)
from byprot.models.dplm2 import (  # noqa: E402
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)

BENCH_DEFAULT = REPO / "data-bin" / "benchmarks" / "mfdesign_nano_vhh"
MF_ROOT = Path(
    "/AIRvePFS/dair/fsq-data/experiments/antibody_design/MFDesign/data/structure_data"
)


def resolve_complex_pdb(entry_id: str) -> Path:
    for split in ("test", "val", "train"):
        cand = MF_ROOT / split / f"{entry_id}.pdb"
        if cand.is_file():
            return cand
    raise FileNotFoundError(entry_id)


def _load_fasta_seqs(path: Path) -> dict[str, str]:
    return {r.id: str(r.seq) for r in SeqIO.parse(str(path), "fasta")}


def prepare_nano_antigen(
    bench_dir: Path,
    struct_tokenizer,
    work_dir: Path,
    device: str,
) -> list[dict]:
    manifest = json.loads((bench_dir / "manifest.json").read_text())
    native = _load_fasta_seqs(bench_dir / "aatype.fasta")
    struct_fasta = _load_fasta_seqs(bench_dir / "struct.fasta")
    rows = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for e in tqdm(manifest["entries"], desc="tokenize_antigen"):
        sid = e["sample_id"]
        eid = e["entry_id"]
        ag_chains = list(e["antigen_chain"])
        pdb_path = resolve_complex_pdb(eid)
        aa_parts, st_parts = [], []
        for cid in ag_chains:
            tok = tokenize_chain_to_row(
                pdb_path,
                cid,
                struct_tokenizer,
                pdb_name=f"{eid}_{cid}",
                work_dir=work_dir,
            )
            aa_parts.append(tok["aa_seq"])
            st_parts.append(tok["struct_seq"])
        ab_aa = native[sid]
        ab_struct = struct_fasta[sid]
        rows.append(
            {
                "sample_id": sid,
                "entry_id": eid,
                "chain_id": e["chain_id"],
                "antigen_chain": e["antigen_chain"],
                "aa_seq": ab_aa,
                "struct_seq": ab_struct,
                "length": len(ab_aa),
                "antigen_aa_seq": "".join(aa_parts),
                "antigen_struct_seq": ",".join(p for p in st_parts if p),
                "pdb_path": str(pdb_path),
            }
        )
        n_ag = len(rows[-1]["antigen_aa_seq"])
        n_st = len([x for x in rows[-1]["antigen_struct_seq"].split(",") if x])
        if n_ag != n_st:
            raise RuntimeError(f"{sid}: ag aa/struct len {n_ag} vs {n_st}")
    return rows


def crop_antigen(
    aa: str,
    struct: str,
    max_len: int,
    epitope_vals: list[int] | None = None,
) -> tuple[str, str, int, int]:
    """Crop Ag aa/struct. Prefer densest-epitope window when labels given.

    Eval uses deterministic leftmost max window (training samples among ties).
    Returns cropped aa, cropped struct csv, start, stop.
    """
    toks = [x for x in struct.split(",") if x]
    n = len(aa)
    if n <= max_len:
        return aa, ",".join(toks), 0, n
    if epitope_vals is not None and len(epitope_vals) == n:
        import numpy as np

        labels = np.asarray(epitope_vals, dtype=np.int32)
        window_sums = np.convolve(
            labels, np.ones(max_len, dtype=np.int32), mode="valid"
        )
        best = int(window_sums.max())
        candidates = np.flatnonzero(window_sums == best)
        start = int(candidates[0])
        stop = start + max_len
    else:
        start = max(0, (n - max_len) // 2)
        stop = start + max_len
    return aa[start:stop], ",".join(toks[start:stop]), start, stop


def _fasta_n(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open() if line.startswith(">"))


def _done(path: Path, n_expected: int) -> bool:
    return _fasta_n(path) >= n_expected


def load_nano43_epitopes(bench_dir: Path) -> dict[str, list[int]]:
    path = bench_dir / "epitope_labels.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {sid: list(v["epitope_mask"]) for sid, v in raw.items()}


def build_antigen_batch(
    tokenizer,
    aa: str,
    struct: str,
    device,
    max_len: int,
    epitope_vals: list[int] | None = None,
):
    aa, struct, start, stop = crop_antigen(aa, struct, max_len, epitope_vals)
    aa_w = tokenizer.aa_cls_token + aa + tokenizer.aa_eos_token
    st_toks = [x for x in struct.split(",") if x]
    st_w = (
        tokenizer.struct_cls_token
        + "".join(st_toks)
        + tokenizer.struct_eos_token
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa_w], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st_w], add_special_tokens=False, return_tensors="pt"
    )
    out = {
        "struct_tokens": {
            "targets": batch_st["input_ids"].to(device),
            "attention_mask": batch_st["attention_mask"].bool().to(device),
        },
        "aatype_tokens": {
            "targets": batch_aa["input_ids"].to(device),
            "attention_mask": batch_aa["attention_mask"].bool().to(device),
        },
        "has_antigen": torch.tensor([True], device=device),
    }
    # Collate format: [0]+cropped_aa+[0] aligned with aa tokens (CLS/EOS).
    if epitope_vals is not None and len(epitope_vals) >= stop:
        cropped = epitope_vals[start:stop]
        labels = [0] + [int(x) for x in cropped] + [0]
        label_t = torch.tensor([labels], dtype=torch.float, device=device)
        valid_t = torch.ones_like(label_t, dtype=torch.bool)
        out["epitope_labels"] = label_t
        out["epitope_mask"] = valid_t
    return out


def build_ab_cogen_tokens(tokenizer, length: int, device):
    # placeholders; generate() remasks all non-special tokens
    aa = (
        tokenizer.aa_cls_token
        + tokenizer.aa_mask_token * length
        + tokenizer.aa_eos_token
    )
    st = (
        tokenizer.struct_cls_token
        + tokenizer.struct_mask_token * length
        + tokenizer.struct_eos_token
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st], add_special_tokens=False, return_tensors="pt"
    )
    tokens = torch.cat(
        [batch_st["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)
    return tokens


def build_ab_invfold_tokens(tokenizer, struct_seq: str, device):
    n = len([x for x in struct_seq.split(",") if x])
    aa = (
        tokenizer.aa_cls_token
        + tokenizer.aa_mask_token * n
        + tokenizer.aa_eos_token
    )
    st = (
        tokenizer.struct_cls_token
        + "".join(struct_seq.split(","))
        + tokenizer.struct_eos_token
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st], add_special_tokens=False, return_tensors="pt"
    )
    input_tokens = torch.cat(
        [batch_st["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)
    type_ids = None
    return input_tokens, n


def build_ab_folding_tokens(tokenizer, aa_seq: str, device):
    n = len(aa_seq)
    aa = tokenizer.aa_cls_token + aa_seq + tokenizer.aa_eos_token
    st = (
        tokenizer.struct_cls_token
        + tokenizer.struct_mask_token * n
        + tokenizer.struct_eos_token
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st], add_special_tokens=False, return_tensors="pt"
    )
    input_tokens = torch.cat(
        [batch_st["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)
    return input_tokens, n


def build_ab_cdr_gen_tokens(
    tokenizer,
    aa_seq: str,
    struct_seq: str,
    cdr_spans: dict,
    device,
):
    """Framework aa+struct kept; CDR1/2/3 aa+struct masked for generation."""
    n = len(aa_seq)
    st_toks = [x for x in struct_seq.split(",") if x]
    if len(st_toks) != n:
        raise ValueError(f"aa/struct length mismatch {n} vs {len(st_toks)}")
    cdr_res = set()
    for key in ("CDR1", "CDR2", "CDR3", "cdr1", "cdr2", "cdr3"):
        if key not in cdr_spans:
            continue
        a, b = cdr_spans[key]
        for i in range(int(a), int(b) + 1):
            if 0 <= i < n:
                cdr_res.add(i)
    if not cdr_res:
        raise ValueError("empty CDR spans")

    aa_chars = []
    st_chars = []
    for i in range(n):
        if i in cdr_res:
            aa_chars.append(tokenizer.aa_mask_token)
            st_chars.append(tokenizer.struct_mask_token)
        else:
            aa_chars.append(aa_seq[i])
            st_chars.append(st_toks[i])
    aa = tokenizer.aa_cls_token + "".join(aa_chars) + tokenizer.aa_eos_token
    st = (
        tokenizer.struct_cls_token
        + "".join(st_chars)
        + tokenizer.struct_eos_token
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st], add_special_tokens=False, return_tensors="pt"
    )
    input_tokens = torch.cat(
        [batch_st["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)
    # partial_masks=True => keep / do not update
    half = n + 2  # cls + residues + eos
    keep = torch.ones(1, 2 * half, dtype=torch.bool, device=device)
    # unmask CDR residue positions on both halves (False = allow generation)
    for i in cdr_res:
        keep[0, 1 + i] = False  # struct half
        keep[0, half + 1 + i] = False  # aa half
    return input_tokens, keep, n


def run_cdr_generation(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
    antigen_max_len: int,
    cdr_regions: dict,
):
    """Generate CDR aa+struct given Ag + Ab framework aa+struct."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="cdr_generation")):
        sid = row["sample_id"]
        if sid not in cdr_regions:
            print(f"[warn] no CDR for {sid}", flush=True)
            continue
        input_tokens, partial_mask, _ = build_ab_cdr_gen_tokens(
            tok,
            row["aa_seq"],
            row["struct_seq"],
            cdr_regions[sid],
            device,
        )
        ag_batch = build_antigen_batch(
            tok,
            row["antigen_aa_seq"],
            row["antigen_struct_seq"],
            device,
            antigen_max_len,
            epitope_vals=row.get("epitope_mask"),
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=ag_batch,
            )
        save_results(
            outputs=outputs,
            task="cdr_generation",
            save_dir=str(out_dir),
            headers=[sid],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=True,
            continue_write=i > 0,
        )


def load_antigen_model(ckpt: Path, stok: Path):
    vocab = REPO / "checkpoints" / "dplm2_650m"
    model = DPLM2.from_pretrained(
        str(ckpt),
        from_huggingface=False,
        cfg_override={
            "tokenizer": {"vocab_file": str(vocab)},
            "struct_tokenizer": {"exp_path": str(stok)},
        },
    )
    # cfg_override may be ignored by from_pretrained local path — ensure stok
    if getattr(model, "struct_tokenizer", None) is None:
        model.struct_tokenizer = get_struct_tokenizer(str(stok))
    model = model.eval().cuda()
    if isinstance(model.net, PeftModel):
        model.net = model.net.merge_and_unload()
    if not getattr(model.cfg.antigen_condition, "enable", False):
        raise RuntimeError(f"{ckpt}: antigen_condition.enable is False")
    if model.antigen_encoder is None:
        raise RuntimeError(f"{ckpt}: missing antigen_encoder")
    return model


def run_cogen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
    antigen_max_len: int,
):
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="co_generation")):
        ab_tokens = build_ab_cogen_tokens(tok, int(row["length"]), device)
        ag_batch = build_antigen_batch(
            tok,
            row["antigen_aa_seq"],
            row["antigen_struct_seq"],
            device,
            antigen_max_len,
            epitope_vals=row.get("epitope_mask"),
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=ab_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=None,
                antigen_batch=ag_batch,
            )
        save_results(
            outputs=outputs,
            task="co_generation",
            save_dir=str(out_dir),
            headers=[row["sample_id"]],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=False,
            continue_write=i > 0,
        )


def run_invfold_antigen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
    antigen_max_len: int,
):
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="invfold+antigen")):
        input_tokens, n = build_ab_invfold_tokens(
            tok, row["struct_seq"], device
        )
        # keep struct half fixed
        type_ids = model.get_modality_type(input_tokens)
        partial_mask = type_ids == model.struct_type
        ag_batch = build_antigen_batch(
            tok,
            row["antigen_aa_seq"],
            row["antigen_struct_seq"],
            device,
            antigen_max_len,
            epitope_vals=row.get("epitope_mask"),
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=ag_batch,
            )
        save_results(
            outputs=outputs,
            task="inverse_folding",
            save_dir=str(out_dir),
            headers=[row["sample_id"]],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=False,
            continue_write=i > 0,
        )


def run_invfold_no_antigen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
):
    """Ab struct only — same as standard invfold, no antigen_batch."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="invfold (no Ag)")):
        input_tokens, n = build_ab_invfold_tokens(
            tok, row["struct_seq"], device
        )
        type_ids = model.get_modality_type(input_tokens)
        partial_mask = type_ids == model.struct_type
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=None,
            )
        save_results(
            outputs=outputs,
            task="inverse_folding",
            save_dir=str(out_dir),
            headers=[row["sample_id"]],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=False,
            continue_write=i > 0,
        )


def run_folding_antigen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
    antigen_max_len: int,
):
    """Ab aa given; generate Ab struct conditioned on antigen aa+struct."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="folding+antigen")):
        input_tokens, _ = build_ab_folding_tokens(tok, row["aa_seq"], device)
        type_ids = model.get_modality_type(input_tokens)
        # keep aa half fixed
        partial_mask = type_ids == model.aa_type
        ag_batch = build_antigen_batch(
            tok,
            row["antigen_aa_seq"],
            row["antigen_struct_seq"],
            device,
            antigen_max_len,
            epitope_vals=row.get("epitope_mask"),
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=ag_batch,
            )
        save_results(
            outputs=outputs,
            task="folding",
            save_dir=str(out_dir),
            headers=[row["sample_id"]],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=True,
            continue_write=i > 0,
        )


def run_folding_no_antigen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
):
    """Ab aa given; generate Ab struct without antigen_batch."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="folding (no Ag)")):
        input_tokens, _ = build_ab_folding_tokens(tok, row["aa_seq"], device)
        type_ids = model.get_modality_type(input_tokens)
        partial_mask = type_ids == model.aa_type
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=None,
            )
        save_results(
            outputs=outputs,
            task="folding",
            save_dir=str(out_dir),
            headers=[row["sample_id"]],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=True,
            continue_write=i > 0,
        )


def _cdr_residue_set(cdr_spans: dict, n: int) -> set[int]:
    cdr_res: set[int] = set()
    for key in ("CDR1", "CDR2", "CDR3", "cdr1", "cdr2", "cdr3"):
        if key not in cdr_spans:
            continue
        a, b = cdr_spans[key]
        for i in range(int(a), int(b) + 1):
            if 0 <= i < n:
                cdr_res.add(i)
    if not cdr_res:
        raise ValueError("empty CDR spans")
    return cdr_res


def build_ab_fr_folding_tokens(
    tokenizer,
    aa_seq: str,
    struct_seq: str,
    cdr_spans: dict,
    device,
):
    """Full Ab aa + framework struct given; only CDR struct positions masked."""
    n = len(aa_seq)
    st_toks = [x for x in struct_seq.split(",") if x]
    if len(st_toks) != n:
        raise ValueError(f"aa/struct length mismatch {n} vs {len(st_toks)}")
    cdr_res = _cdr_residue_set(cdr_spans, n)
    st_chars = [
        tokenizer.struct_mask_token if i in cdr_res else st_toks[i]
        for i in range(n)
    ]
    aa = tokenizer.aa_cls_token + aa_seq + tokenizer.aa_eos_token
    st = (
        tokenizer.struct_cls_token
        + "".join(st_chars)
        + tokenizer.struct_eos_token
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st], add_special_tokens=False, return_tensors="pt"
    )
    input_tokens = torch.cat(
        [batch_st["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)
    half = n + 2
    keep = torch.ones(1, 2 * half, dtype=torch.bool, device=device)
    for i in cdr_res:
        keep[0, 1 + i] = False  # only generate CDR struct
    return input_tokens, keep, n


def build_ab_fr_invfold_tokens(
    tokenizer,
    aa_seq: str,
    struct_seq: str,
    cdr_spans: dict,
    device,
):
    """Full Ab struct + framework aa given; only CDR aa positions masked."""
    n = len(aa_seq)
    st_toks = [x for x in struct_seq.split(",") if x]
    if len(st_toks) != n:
        raise ValueError(f"aa/struct length mismatch {n} vs {len(st_toks)}")
    cdr_res = _cdr_residue_set(cdr_spans, n)
    aa_chars = [
        tokenizer.aa_mask_token if i in cdr_res else aa_seq[i] for i in range(n)
    ]
    aa = tokenizer.aa_cls_token + "".join(aa_chars) + tokenizer.aa_eos_token
    st = tokenizer.struct_cls_token + "".join(st_toks) + tokenizer.struct_eos_token
    batch_aa = tokenizer.batch_encode_plus(
        [aa], add_special_tokens=False, return_tensors="pt"
    )
    batch_st = tokenizer.batch_encode_plus(
        [st], add_special_tokens=False, return_tensors="pt"
    )
    input_tokens = torch.cat(
        [batch_st["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)
    half = n + 2
    keep = torch.ones(1, 2 * half, dtype=torch.bool, device=device)
    for i in cdr_res:
        keep[0, half + 1 + i] = False  # only generate CDR aa
    return input_tokens, keep, n


def run_fr_folding_antigen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
    antigen_max_len: int,
    cdr_regions: dict,
):
    """Given Ag + Ab aa + framework struct; generate CDR structures only."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="fr_folding+antigen")):
        sid = row["sample_id"]
        if sid not in cdr_regions:
            print(f"[warn] no CDR for {sid}", flush=True)
            continue
        input_tokens, partial_mask, _ = build_ab_fr_folding_tokens(
            tok,
            row["aa_seq"],
            row["struct_seq"],
            cdr_regions[sid],
            device,
        )
        ag_batch = build_antigen_batch(
            tok,
            row["antigen_aa_seq"],
            row["antigen_struct_seq"],
            device,
            antigen_max_len,
            epitope_vals=row.get("epitope_mask"),
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=ag_batch,
            )
        save_results(
            outputs=outputs,
            task="folding",
            save_dir=str(out_dir),
            headers=[sid],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=True,
            continue_write=i > 0,
        )


def run_fr_invfold_antigen(
    model,
    rows: list[dict],
    out_dir: Path,
    max_iter: int,
    antigen_max_len: int,
    cdr_regions: dict,
):
    """Given Ag + Ab struct + framework aa; generate CDR sequences only."""
    device = next(model.parameters()).device
    tok = model.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(tqdm(rows, desc="fr_invfold+antigen")):
        sid = row["sample_id"]
        if sid not in cdr_regions:
            print(f"[warn] no CDR for {sid}", flush=True)
            continue
        input_tokens, partial_mask, _ = build_ab_fr_invfold_tokens(
            tok,
            row["aa_seq"],
            row["struct_seq"],
            cdr_regions[sid],
            device,
        )
        ag_batch = build_antigen_batch(
            tok,
            row["antigen_aa_seq"],
            row["antigen_struct_seq"],
            device,
            antigen_max_len,
            epitope_vals=row.get("epitope_mask"),
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
                antigen_batch=ag_batch,
            )
        save_results(
            outputs=outputs,
            task="inverse_folding",
            save_dir=str(out_dir),
            headers=[sid],
            tokenizer=tok,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=False,
            continue_write=i > 0,
        )


def score_folding(pred_pdb_dir: Path, gt_pdb_dir: Path, cdr: dict | None) -> dict:
    rows = compute_folding_metrics(pred_pdb_dir, gt_pdb_dir, cdr_regions=cdr)
    ok = [r for r in rows if r.get("status") == "ok"]
    ca = summarize_metrics(ok, "ca_rmsd")
    tm = summarize_metrics(ok, "bb_tmscore")
    regs = summarize_region_metrics(ok, "ca_rmsd") if cdr else {}
    return {
        "n": ca.get("count"),
        "ca_rmsd": ca.get("mean"),
        "bb_tmscore": tm.get("mean"),
        "regions_ca_rmsd": {k: v.get("mean") for k, v in regs.items()},
        "rows": rows,
    }


def score_aar(
    pred_fasta: Path,
    native: dict[str, str],
    cdr: dict | None,
) -> dict:
    rows = compute_inverse_folding_metrics(pred_fasta, native, cdr)
    glob = summarize_metrics(rows, "seq_recovery")
    regs = summarize_region_metrics(rows, "seq_recovery") if cdr else {}
    return {
        "n": glob.get("count"),
        "global_aar": glob.get("mean"),
        "regions": {k: v.get("mean") for k, v in regs.items()},
        "rows": rows,
    }


def pct(x):
    return f"{100.0 * x:.1f}" if x is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DEFAULT)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "outputs" / "vhh_eval_antigen_cond_nano43",
    )
    ap.add_argument("--antigen-max-len", type=int, default=384)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument(
        "--modes",
        type=str,
        default="co_generation,inverse_folding",
        help=(
            "comma: co_generation, inverse_folding, inverse_folding_no_ag, folding, folding_no_ag, "
            "cdr_generation, fr_folding, fr_inverse_folding"
        ),
    )
    ap.add_argument(
        "--ckpts",
        nargs="*",
        default=None,
        help="tag=path pairs; default = saved antigen_cond ckpts",
    )
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    require_cuda()
    stok = REPO / "checkpoints" / "struct_tokenizer"
    cache_json = args.bench_dir / "antigen_eval_rows.json"
    cache_csv = args.bench_dir / "antigen_eval_rows.csv"

    if cache_json.is_file() and not args.prepare_only:
        rows = json.loads(cache_json.read_text())
        print(f"loaded cache n={len(rows)}", flush=True)
    else:
        print("preparing nano43 antigen tokens...", flush=True)
        struct_tokenizer = get_struct_tokenizer(str(stok))
        if torch.cuda.is_available():
            struct_tokenizer = struct_tokenizer.cuda().eval()
        rows = prepare_nano_antigen(
            args.bench_dir,
            struct_tokenizer,
            args.bench_dir / "_antigen_chain_pdbs",
            "cuda",
        )
        cache_json.write_text(json.dumps(rows, indent=2))
        import pandas as pd

        pd.DataFrame(rows).to_csv(cache_csv, index=False)
        print(f"wrote {cache_json} n={len(rows)}", flush=True)
        del struct_tokenizer
        torch.cuda.empty_cache()
        if args.prepare_only:
            return

    epitopes = load_nano43_epitopes(args.bench_dir)
    if epitopes:
        n_attach = 0
        for r in rows:
            epi = epitopes.get(r["sample_id"])
            if epi is not None:
                r["epitope_mask"] = epi
                n_attach += 1
        print(
            f"attached epitope masks: {n_attach}/{len(rows)} "
            f"(from {args.bench_dir / 'epitope_labels.json'})",
            flush=True,
        )
    else:
        print(
            f"[warn] no epitope_labels.json under {args.bench_dir}; "
            "cross-attn epitope mask will not apply at eval",
            flush=True,
        )

    if args.ckpts:
        models = []
        for item in args.ckpts:
            tag, path = item.split("=", 1)
            models.append((tag, Path(path)))
    else:
        models = [
            (
                "s1_step500",
                REPO
                / "logs/dplm2_vhh_mfdesign_antigen_cond/checkpoints/every500/step_500.ckpt",
            ),
            (
                "s1_step1000",
                REPO
                / "logs/dplm2_vhh_mfdesign_antigen_cond/checkpoints/every500/step_1000.ckpt",
            ),
            (
                "s2_step500",
                REPO
                / "logs/dplm2_vhh_mfdesign_antigen_cond_s2/checkpoints/every500/step_500.ckpt",
            ),
            (
                "s2_step1000",
                REPO
                / "logs/dplm2_vhh_mfdesign_antigen_cond_s2/checkpoints/every500/step_1000.ckpt",
            ),
        ]

    native = _load_fasta_seqs(args.bench_dir / "aatype.fasta")
    cdr_raw = json.loads(
        (args.bench_dir / "cdr_regions_anarci_chothia.json").read_text()
    )
    cdr = to_lib_regions(cdr_raw)

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for tag, ckpt in models:
        if not ckpt.is_file():
            print(f"[skip] missing {ckpt}", flush=True)
            continue
        run_dir = args.out_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {"tag": tag, "ckpt": str(ckpt)}

        if not args.metrics_only:
            print(f"[load] {tag} <- {ckpt}", flush=True)
            model = load_antigen_model(ckpt, stok)
            if "co_generation" in modes:
                cogen_dir = run_dir / "co_generation"
                if _done(cogen_dir / "aatype.fasta", len(rows)):
                    print(f"[skip] {tag}/co_generation already complete", flush=True)
                else:
                    if (cogen_dir / "aatype.fasta").is_file():
                        (cogen_dir / "aatype.fasta").unlink()
                        st = cogen_dir / "struct_token.fasta"
                        if st.is_file():
                            st.unlink()
                    run_cogen(
                        model,
                        rows,
                        cogen_dir,
                        args.max_iter,
                        args.antigen_max_len,
                    )
            if "inverse_folding" in modes:
                inv_dir = run_dir / "inverse_folding"
                if _done(inv_dir / "aatype.fasta", len(rows)):
                    print(f"[skip] {tag}/inverse_folding already complete", flush=True)
                else:
                    if (inv_dir / "aatype.fasta").is_file():
                        (inv_dir / "aatype.fasta").unlink()
                        st = inv_dir / "struct_token.fasta"
                        if st.is_file():
                            st.unlink()
                    run_invfold_antigen(
                        model,
                        rows,
                        inv_dir,
                        args.max_iter,
                        args.antigen_max_len,
                    )
            if "inverse_folding_no_ag" in modes:
                inv_dir = run_dir / "inverse_folding_no_ag"
                if _done(inv_dir / "aatype.fasta", len(rows)):
                    print(
                        f"[skip] {tag}/inverse_folding_no_ag already complete",
                        flush=True,
                    )
                else:
                    if (inv_dir / "aatype.fasta").is_file():
                        (inv_dir / "aatype.fasta").unlink()
                        st = inv_dir / "struct_token.fasta"
                        if st.is_file():
                            st.unlink()
                    run_invfold_no_antigen(
                        model,
                        rows,
                        inv_dir,
                        args.max_iter,
                    )
            if "folding" in modes:
                fold_dir = run_dir / "folding"
                n_pdb = len(list((fold_dir / "pdb").glob("*.pdb"))) if (fold_dir / "pdb").is_dir() else 0
                if n_pdb >= len(rows):
                    print(f"[skip] {tag}/folding already complete ({n_pdb})", flush=True)
                else:
                    if (fold_dir / "aatype.fasta").is_file():
                        (fold_dir / "aatype.fasta").unlink()
                    st = fold_dir / "struct_token.fasta"
                    if st.is_file():
                        st.unlink()
                    pdb_dir = fold_dir / "pdb"
                    if pdb_dir.is_dir():
                        for p in pdb_dir.glob("*.pdb"):
                            p.unlink()
                    run_folding_antigen(
                        model,
                        rows,
                        fold_dir,
                        args.max_iter,
                        args.antigen_max_len,
                    )
            if "folding_no_ag" in modes:
                fold_dir = run_dir / "folding_no_ag"
                n_pdb = (
                    len(list((fold_dir / "pdb").glob("*.pdb")))
                    if (fold_dir / "pdb").is_dir()
                    else 0
                )
                if n_pdb >= len(rows):
                    print(
                        f"[skip] {tag}/folding_no_ag already complete ({n_pdb})",
                        flush=True,
                    )
                else:
                    if (fold_dir / "aatype.fasta").is_file():
                        (fold_dir / "aatype.fasta").unlink()
                    st = fold_dir / "struct_token.fasta"
                    if st.is_file():
                        st.unlink()
                    pdb_dir = fold_dir / "pdb"
                    if pdb_dir.is_dir():
                        for p in pdb_dir.glob("*.pdb"):
                            p.unlink()
                    run_folding_no_antigen(
                        model,
                        rows,
                        fold_dir,
                        args.max_iter,
                    )
            if "cdr_generation" in modes:
                cdr_dir = run_dir / "cdr_generation"
                n_pdb = (
                    len(list((cdr_dir / "pdb").glob("*.pdb")))
                    if (cdr_dir / "pdb").is_dir()
                    else 0
                )
                if n_pdb >= len(rows):
                    print(
                        f"[skip] {tag}/cdr_generation already complete ({n_pdb})",
                        flush=True,
                    )
                else:
                    if (cdr_dir / "aatype.fasta").is_file():
                        (cdr_dir / "aatype.fasta").unlink()
                    st = cdr_dir / "struct_token.fasta"
                    if st.is_file():
                        st.unlink()
                    pdb_dir = cdr_dir / "pdb"
                    if pdb_dir.is_dir():
                        for p in pdb_dir.glob("*.pdb"):
                            p.unlink()
                    run_cdr_generation(
                        model,
                        rows,
                        cdr_dir,
                        args.max_iter,
                        args.antigen_max_len,
                        cdr,
                    )
            if "fr_folding" in modes:
                frf_dir = run_dir / "fr_folding"
                n_pdb = (
                    len(list((frf_dir / "pdb").glob("*.pdb")))
                    if (frf_dir / "pdb").is_dir()
                    else 0
                )
                if n_pdb >= len(rows):
                    print(f"[skip] {tag}/fr_folding already complete ({n_pdb})", flush=True)
                else:
                    if (frf_dir / "aatype.fasta").is_file():
                        (frf_dir / "aatype.fasta").unlink()
                    st = frf_dir / "struct_token.fasta"
                    if st.is_file():
                        st.unlink()
                    pdb_dir = frf_dir / "pdb"
                    if pdb_dir.is_dir():
                        for p in pdb_dir.glob("*.pdb"):
                            p.unlink()
                    run_fr_folding_antigen(
                        model,
                        rows,
                        frf_dir,
                        args.max_iter,
                        args.antigen_max_len,
                        cdr,
                    )
            if "fr_inverse_folding" in modes:
                fri_dir = run_dir / "fr_inverse_folding"
                if _done(fri_dir / "aatype.fasta", len(rows)):
                    print(f"[skip] {tag}/fr_inverse_folding already complete", flush=True)
                else:
                    if (fri_dir / "aatype.fasta").is_file():
                        (fri_dir / "aatype.fasta").unlink()
                    st = fri_dir / "struct_token.fasta"
                    if st.is_file():
                        st.unlink()
                    run_fr_invfold_antigen(
                        model,
                        rows,
                        fri_dir,
                        args.max_iter,
                        args.antigen_max_len,
                        cdr,
                    )
            del model
            torch.cuda.empty_cache()

        gt_pdb = args.bench_dir / "pdb_h"
        for mode in modes:
            if mode == "folding":
                pred_pdb = run_dir / "folding" / "pdb"
                if not pred_pdb.is_dir() or not any(pred_pdb.glob("*.pdb")):
                    print(f"[warn] no pred PDB for {tag}/folding", flush=True)
                    continue
                scored = score_folding(pred_pdb, gt_pdb, cdr)
                summary["folding"] = {
                    "n": scored["n"],
                    "ca_rmsd": scored["ca_rmsd"],
                    "bb_tmscore": scored["bb_tmscore"],
                    "regions_ca_rmsd": scored["regions_ca_rmsd"],
                }
                with (run_dir / "folding" / "per_sample.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(scored["rows"][0].keys()))
                    w.writeheader()
                    w.writerows(scored["rows"])
                print(
                    f"[{tag}/folding] CA_RMSD={scored['ca_rmsd']:.2f}Å "
                    f"TM={scored['bb_tmscore']:.3f} N={scored['n']} "
                    f"CDR3={scored['regions_ca_rmsd'].get('CDR3')}",
                    flush=True,
                )
                continue
            if mode == "folding_no_ag":
                pred_pdb = run_dir / "folding_no_ag" / "pdb"
                if not pred_pdb.is_dir() or not any(pred_pdb.glob("*.pdb")):
                    print(f"[warn] no pred PDB for {tag}/folding_no_ag", flush=True)
                    continue
                scored = score_folding(pred_pdb, gt_pdb, cdr)
                summary["folding_no_ag"] = {
                    "n": scored["n"],
                    "ca_rmsd": scored["ca_rmsd"],
                    "bb_tmscore": scored["bb_tmscore"],
                    "regions_ca_rmsd": scored["regions_ca_rmsd"],
                }
                with (run_dir / "folding_no_ag" / "per_sample.csv").open(
                    "w", newline=""
                ) as f:
                    w = csv.DictWriter(f, fieldnames=list(scored["rows"][0].keys()))
                    w.writeheader()
                    w.writerows(scored["rows"])
                print(
                    f"[{tag}/folding_no_ag] CA_RMSD={scored['ca_rmsd']:.2f}Å "
                    f"TM={scored['bb_tmscore']:.3f} N={scored['n']} "
                    f"CDR3={scored['regions_ca_rmsd'].get('CDR3')}",
                    flush=True,
                )
                continue
            if mode == "fr_folding":
                pred_pdb = run_dir / "fr_folding" / "pdb"
                if not pred_pdb.is_dir() or not any(pred_pdb.glob("*.pdb")):
                    print(f"[warn] no pred PDB for {tag}/fr_folding", flush=True)
                    continue
                scored = score_folding(pred_pdb, gt_pdb, cdr)
                summary["fr_folding"] = {
                    "n": scored["n"],
                    "ca_rmsd": scored["ca_rmsd"],
                    "bb_tmscore": scored["bb_tmscore"],
                    "regions_ca_rmsd": scored["regions_ca_rmsd"],
                }
                with (run_dir / "fr_folding" / "per_sample.csv").open(
                    "w", newline=""
                ) as f:
                    w = csv.DictWriter(f, fieldnames=list(scored["rows"][0].keys()))
                    w.writeheader()
                    w.writerows(scored["rows"])
                print(
                    f"[{tag}/fr_folding] CA_RMSD={scored['ca_rmsd']:.2f}Å "
                    f"TM={scored['bb_tmscore']:.3f} N={scored['n']} "
                    f"Fr={scored['regions_ca_rmsd'].get('Fr')} "
                    f"CDR3={scored['regions_ca_rmsd'].get('CDR3')}",
                    flush=True,
                )
                continue
            if mode == "cdr_generation":
                pred = run_dir / mode / "aatype.fasta"
                if pred.is_file():
                    scored = score_aar(pred, native, cdr)
                    summary[mode] = {
                        "n": scored["n"],
                        "global_aar": scored["global_aar"],
                        "regions": scored["regions"],
                    }
                    with (run_dir / mode / "per_sample_aar.csv").open(
                        "w", newline=""
                    ) as f:
                        w = csv.DictWriter(
                            f, fieldnames=list(scored["rows"][0].keys())
                        )
                        w.writeheader()
                        w.writerows(scored["rows"])
                    print(
                        f"[{tag}/{mode}] Global={pct(scored['global_aar'])} "
                        f"Fr={pct(scored['regions'].get('Fr'))} "
                        f"CDR3={pct(scored['regions'].get('CDR3'))} N={scored['n']}",
                        flush=True,
                    )
                pred_pdb = run_dir / mode / "pdb"
                if pred_pdb.is_dir() and any(pred_pdb.glob("*.pdb")):
                    fscored = score_folding(pred_pdb, gt_pdb, cdr)
                    summary[mode] = summary.get(mode, {})
                    summary[mode].update(
                        {
                            "ca_rmsd": fscored["ca_rmsd"],
                            "bb_tmscore": fscored["bb_tmscore"],
                            "regions_ca_rmsd": fscored["regions_ca_rmsd"],
                        }
                    )
                    print(
                        f"[{tag}/{mode}] CA_RMSD={fscored['ca_rmsd']:.2f}Å "
                        f"CDR3_RMSD={fscored['regions_ca_rmsd'].get('CDR3')}",
                        flush=True,
                    )
                continue
            pred = run_dir / mode / "aatype.fasta"
            if not pred.is_file():
                print(f"[warn] no pred for {tag}/{mode}", flush=True)
                continue
            scored = score_aar(pred, native, cdr)
            summary[mode] = {
                "n": scored["n"],
                "global_aar": scored["global_aar"],
                "regions": scored["regions"],
            }
            with (run_dir / mode / "per_sample.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(scored["rows"][0].keys()))
                w.writeheader()
                w.writerows(scored["rows"])
            print(
                f"[{tag}/{mode}] Global={pct(scored['global_aar'])} "
                f"Fr={pct(scored['regions'].get('Fr'))} "
                f"CDR3={pct(scored['regions'].get('CDR3'))} N={scored['n']}",
                flush=True,
            )
            print_region_table(
                f"{tag}/{mode}",
                {
                    k: {"mean": v}
                    for k, v in scored["regions"].items()
                    if v is not None
                },
            )
        all_summaries.append(summary)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # comparison markdown
    lines = [
        "# Antigen-conditioned Ab generation on MFDesign nano43",
        "",
        "Given antigen aa+struct; generate antibody "
        "(co_generation = aa+struct; inverse_folding = aa given Ab struct + antigen; "
        "folding = Ab struct given Ab aa + antigen; "
        "cdr_generation = framework fixed, generate CDR1–3 aa+struct; "
        "fr_folding = Fr aa+struct + full aa given, generate CDR struct; "
        "fr_inverse_folding = Fr aa + full struct given, generate CDR aa).",
        "",
    ]
    for mode in modes:
        if mode in ("folding", "fr_folding"):
            lines.append(f"## {mode}")
            lines.append("")
            lines.append(
                "| Model | Mean CA RMSD (Å) | Mean TM | Fr RMSD | CDR1 | CDR2 | **CDR3** | N |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for s in all_summaries:
                m = s.get(mode) or {}
                regs = m.get("regions_ca_rmsd") or {}
                ca = m.get("ca_rmsd")
                tm = m.get("bb_tmscore")
                lines.append(
                    "| {tag} | {ca} | {tm} | {fr} | {c1} | {c2} | {c3} | {n} |".format(
                        tag=s["tag"],
                        ca=f"{ca:.2f}" if ca is not None else "—",
                        tm=f"{tm:.3f}" if tm is not None else "—",
                        fr=f"{regs['Fr']:.2f}" if regs.get("Fr") is not None else "—",
                        c1=f"{regs['CDR1']:.2f}" if regs.get("CDR1") is not None else "—",
                        c2=f"{regs['CDR2']:.2f}" if regs.get("CDR2") is not None else "—",
                        c3=f"{regs['CDR3']:.2f}" if regs.get("CDR3") is not None else "—",
                        n=m.get("n") or "—",
                    )
                )
            lines.append("")
            continue
        lines.append(f"## {mode}")
        lines.append("")
        lines.append(
            "| Model | Global AAR % | Fr % | CDR1 % | CDR2 % | **CDR3 %** | N |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s in all_summaries:
            m = s.get(mode) or {}
            regs = m.get("regions") or {}
            lines.append(
                "| {tag} | {g} | {fr} | {c1} | {c2} | {c3} | {n} |".format(
                    tag=s["tag"],
                    g=pct(m.get("global_aar")),
                    fr=pct(regs.get("Fr")),
                    c1=pct(regs.get("CDR1")),
                    c2=pct(regs.get("CDR2")),
                    c3=pct(regs.get("CDR3")),
                    n=m.get("n") or "—",
                )
            )
        lines.append("")
        if mode == "cdr_generation":
            lines.append(
                "| Model | Mean CA RMSD (Å) | Fr RMSD | CDR1 | CDR2 | **CDR3** | N |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for s in all_summaries:
                m = s.get(mode) or {}
                regs = m.get("regions_ca_rmsd") or {}
                ca = m.get("ca_rmsd")
                lines.append(
                    "| {tag} | {ca} | {fr} | {c1} | {c2} | {c3} | {n} |".format(
                        tag=s["tag"],
                        ca=f"{ca:.2f}" if ca is not None else "—",
                        fr=f"{regs['Fr']:.2f}" if regs.get("Fr") is not None else "—",
                        c1=f"{regs['CDR1']:.2f}" if regs.get("CDR1") is not None else "—",
                        c2=f"{regs['CDR2']:.2f}" if regs.get("CDR2") is not None else "—",
                        c3=f"{regs['CDR3']:.2f}" if regs.get("CDR3") is not None else "—",
                        n=m.get("n") or "—",
                    )
                )
            lines.append("")
    md = args.out_root / "comparison.md"
    md.write_text("\n".join(lines))
    (args.out_root / "comparison.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    print(f"wrote {md}", flush=True)


if __name__ == "__main__":
    main()
