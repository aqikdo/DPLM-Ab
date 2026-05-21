"""Shared helpers for DPLM-2 smoke tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root: tests/smoke -> tests -> dplm
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = REPO_ROOT / "checkpoints" / "dplm2_150m"
DEFAULT_STRUCT_TOKENIZER = REPO_ROOT / "checkpoints" / "struct_tokenizer"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "smoke_outputs"
BENCHMARK_ROOT = REPO_ROOT / "data-bin"
CAMEO_DIR = BENCHMARK_ROOT / "cameo2022"
CAMEO_MINI_DIR = BENCHMARK_ROOT / "cameo2022_mini"


def resolve_model_path(model: str | Path | None) -> Path:
    path = Path(model or os.environ.get("DPLM2_MODEL", DEFAULT_MODEL))
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Download with: huggingface-cli download airkingbd/dplm2_150m "
            f"--local-dir {DEFAULT_MODEL}"
        )
    if not (path / "pytorch_model.bin").exists() and not (path / "config.json").exists():
        raise FileNotFoundError(f"Invalid model dir (missing weights): {path}")
    return path.resolve()


def resolve_struct_tokenizer(path: str | Path | None) -> Path:
    stok = Path(
        path or os.environ.get("DPLM2_STRUCT_TOKENIZER", DEFAULT_STRUCT_TOKENIZER)
    )
    hydra_cfg = stok / ".hydra" / "config.yaml"
    hydra_ckpt = stok / ".hydra" / "dplm2_struct_tokenizer.ckpt"
    if hydra_cfg.exists() and hydra_ckpt.exists():
        return stok.resolve()
    # HF snapshot layout (flat files)
    if (stok / "config.yaml").exists() and (stok / "dplm2_struct_tokenizer.ckpt").exists():
        return stok.resolve()
    raise FileNotFoundError(
        f"Struct tokenizer not found: {stok}\n"
        "Expected .hydra/config.yaml and .hydra/dplm2_struct_tokenizer.ckpt"
    )


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPLM-2 smoke tests.")


def load_dplm2(model_path: str | Path | None = None, struct_tokenizer_path: str | Path | None = None):
    import torch
    from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2

    model_dir = resolve_model_path(model_path)
    stok_dir = resolve_struct_tokenizer(struct_tokenizer_path)

    model = DPLM2.from_pretrained(
        str(model_dir),
        cfg_override={
            "tokenizer": {"vocab_file": str(model_dir)},
            "struct_tokenizer": {"exp_path": str(stok_dir)},
        },
    )
    model = model.eval().cuda()
    if hasattr(model.net, "merge_and_unload"):
        from peft.peft_model import PeftModel

        if isinstance(model.net, PeftModel):
            model.net = model.net.merge_and_unload()
    return model, model_dir, stok_dir


def benchmark_available() -> bool:
    return (CAMEO_MINI_DIR / "aatype.fasta").exists() and (
        CAMEO_MINI_DIR / "struct.fasta"
    ).exists()


def benchmark_metadata_available() -> bool:
    return (BENCHMARK_ROOT / "metadata" / "pdb_afdb_cameo.csv").exists()
