# Minimal helpers for antibody smoke tests (no src/byprot dependency).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

SCFV_LINKER_AA = "GGGGSGGGGSGGGGS"
SCFV_LINKER_LEN = len(SCFV_LINKER_AA)

CHOTHIA_VH_CDR = {
    "cdr_h1": (26, 32),
    "cdr_h2": (52, 56),
    "cdr_h3": (95, 102),
}
CHOTHIA_VL_CDR = {
    "cdr_l1": (24, 34),
    "cdr_l2": (50, 56),
    "cdr_l3": (89, 97),
}


def _set_range(mask: np.ndarray, start: int, end: int) -> None:
    mask[start - 1 : end] = True


def generate_antibody_masks(
    vh_len: int,
    vl_len: int,
    linker_len: int = SCFV_LINKER_LEN,
) -> Dict[str, np.ndarray]:
    total = vh_len + linker_len + vl_len
    cdr_mask = np.zeros(total, dtype=bool)
    for start, end in CHOTHIA_VH_CDR.values():
        _set_range(cdr_mask, start, end)
    vl_offset = vh_len + linker_len
    for start, end in CHOTHIA_VL_CDR.values():
        _set_range(cdr_mask, start + vl_offset, end + vl_offset)

    partial_mask = ~cdr_mask
    loss_mask = cdr_mask.copy()
    struct_mask = cdr_mask.copy()
    struct_mask[vh_len : vh_len + linker_len] = True
    return {
        "cdr_mask": cdr_mask,
        "partial_mask": partial_mask,
        "loss_mask": loss_mask,
        "struct_mask": struct_mask,
    }


@dataclass
class ScFvRecord:
    pdb_id: str
    scfv_sequence: str
    vh_len: int
    vl_len: int
    cdr_mask: np.ndarray


def build_scfv_record(pdb_id: str, vh_sequence: str, vl_sequence: str) -> ScFvRecord:
    scfv_seq = vh_sequence + SCFV_LINKER_AA + vl_sequence
    masks = generate_antibody_masks(len(vh_sequence), len(vl_sequence))
    return ScFvRecord(
        pdb_id=pdb_id,
        scfv_sequence=scfv_seq,
        vh_len=len(vh_sequence),
        vl_len=len(vl_sequence),
        cdr_mask=masks["cdr_mask"],
    )


def split_scfv_to_chains(record: ScFvRecord) -> Tuple[str, str, Tuple[int, int, int]]:
    return (
        record.scfv_sequence[: record.vh_len],
        record.scfv_sequence[record.vh_len + SCFV_LINKER_LEN :],
        (record.vh_len, SCFV_LINKER_LEN, record.vl_len),
    )


def compute_aar(pred_seq: str, native_seq: str, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        return sum(pred_seq[i] == native_seq[i] for i in idx) / len(idx)
    return sum(a == b for a, b in zip(pred_seq, native_seq)) / len(native_seq)


def compute_cdr_rmsd(
    pred_ca: np.ndarray,
    native_ca: np.ndarray,
    framework_mask: np.ndarray,
    cdr_mask: np.ndarray,
) -> Tuple[float, float]:
    fw_idx = np.where(framework_mask)[0]
    cdr_idx = np.where(cdr_mask)[0]
    if len(fw_idx) < 3 or len(cdr_idx) == 0:
        return float("nan"), float("nan")
    mobile = pred_ca[fw_idx] - pred_ca[fw_idx].mean(0)
    target = native_ca[fw_idx] - native_ca[fw_idx].mean(0)
    c = mobile.T @ target
    v, _, w = np.linalg.svd(c)
    d = np.sign(np.linalg.det(v @ w))
    rot = v @ np.diag([1, 1, d]) @ w
    pred_aligned = (pred_ca - pred_ca[fw_idx].mean(0)) @ rot + native_ca[fw_idx].mean(0)
    diff = pred_aligned[cdr_idx] - native_ca[cdr_idx]
    rmsd = float(np.sqrt((diff**2).sum() / len(cdr_idx)))
    return rmsd, rmsd


def split_scfv_pdb(
    ca_coords: np.ndarray, vh_len: int, linker_len: int, vl_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    vh = ca_coords[:vh_len]
    vl = ca_coords[vh_len + linker_len : vh_len + linker_len + vl_len]
    return vh, vl
