#!/usr/bin/env python3
"""Smoke tests for antibody scFv masks, zero-init cross-attn, and metrics (self-contained)."""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

SMOKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SMOKE_DIR))

from _antibody_smoke_lib import (  # noqa: E402
    SCFV_LINKER_AA,
    build_scfv_record,
    compute_aar,
    compute_cdr_rmsd,
    generate_antibody_masks,
    split_scfv_pdb,
    split_scfv_to_chains,
)


class ZeroInitCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_antibody: torch.Tensor, h_antigen: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=x_antibody, key=h_antigen, value=h_antigen
        )
        return self.out_proj(attn_out)


def test_scfv_masks():
    vh_len, vl_len = 120, 110
    masks = generate_antibody_masks(vh_len, vl_len)
    total = vh_len + len(SCFV_LINKER_AA) + vl_len
    assert masks["cdr_mask"].shape == (total,)
    assert masks["partial_mask"].sum() > masks["cdr_mask"].sum()
    linker_slice = slice(vh_len, vh_len + len(SCFV_LINKER_AA))
    assert masks["struct_mask"][linker_slice].all()
    assert not masks["loss_mask"][linker_slice].any()
    rec = build_scfv_record("1abc", "A" * vh_len, "C" * vl_len)
    assert len(rec.scfv_sequence) == total
    vh, vl, lens = split_scfv_to_chains(rec)
    assert lens == (vh_len, len(SCFV_LINKER_AA), vl_len)
    assert len(vh) == vh_len and len(vl) == vl_len
    print("test_scfv_masks OK")


def test_zero_init_cross_attention():
    d, heads, B, L_ab, L_ag = 64, 4, 2, 20, 15
    mod = ZeroInitCrossAttention(d, heads)
    x = torch.randn(B, L_ab, d)
    h = torch.randn(B, L_ag, d)
    out = mod(x, h)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    print("test_zero_init_cross_attention OK")


def test_metrics():
    native = "ARNDCQEGHILKMFPSTWYV" * 5
    pred = list(native)
    pred[0] = "X"
    mask = np.zeros(len(native), dtype=bool)
    mask[:10] = True
    aar = compute_aar("".join(pred), native, mask=mask)
    assert 0.0 <= aar <= 1.0
    L = 115
    pred_ca = np.random.randn(L, 3).astype(np.float32)
    native_ca = pred_ca + 0.1 * np.random.randn(L, 3).astype(np.float32)
    cdr_mask = np.zeros(L, dtype=bool)
    cdr_mask[50:60] = True
    fw_mask = ~cdr_mask
    rmsd, _ = compute_cdr_rmsd(pred_ca, native_ca, fw_mask, cdr_mask)
    assert rmsd >= 0
    vh, vl = split_scfv_pdb(pred_ca, 50, 15, 50)
    assert vh.shape[0] == 50 and vl.shape[0] == 50
    print("test_metrics OK")


if __name__ == "__main__":
    test_scfv_masks()
    test_zero_init_cross_attention()
    test_metrics()
    print("All antibody module smoke tests passed.")
