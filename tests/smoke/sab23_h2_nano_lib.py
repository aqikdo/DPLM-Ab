"""Utilities for SAb-23-H2-Nano single-chain nanobody benchmark (DPLM-2 folding / inverse folding)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from openfold.utils.superimposition import superimpose

from byprot.datamodules.pdb_dataset import utils as du
from byprot.utils.protein.utils import calc_tm_score

from common import REPO_ROOT

DEFAULT_DATASET_DIR = Path(
    os.environ.get("SAB23_H2_NANO_DIR", Path.home() / "SAb-23-H2-Nano")
)
BENCHMARK_DIR = REPO_ROOT / "data-bin" / "sab23_h2_nano"

DEFAULT_IGFOLD_BENCHMARK_ROOT = Path(
    os.environ.get("IGFOLD_BENCHMARK_ROOT", Path.home() / "benchmark")
)
DEFAULT_IGFOLD_XTAL_DIR = Path(
    os.environ.get(
        "IGFOLD_XTAL_DIR",
        Path.home() / "IgFold_benchmark" / "xtal" / "July2021_nano",
    )
)
IGFOLD_BENCHMARK_DIR = REPO_ROOT / "data-bin" / "igfold_nano"

NANOBODY_CHAIN = "H"
REGION_NAMES = ("Fr", "CDR1", "CDR2", "CDR3")

# Chothia VH (1-based inclusive); IgFold benchmark uses Chothia-renumbered Fv structures.
CHOTHIA_VH_CDR1 = (26, 32)
CHOTHIA_VH_CDR2 = (52, 56)
CHOTHIA_VH_CDR3_START = 95
CHOTHIA_VH_CDR3_DEFAULT_LEN = 8


class _ChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id


def list_sample_ids(dataset_dir: Path) -> list[str]:
    native_fasta = dataset_dir / "fasta.files.native"
    if not native_fasta.is_dir():
        raise FileNotFoundError(f"Missing {native_fasta}")
    return sorted(p.stem for p in native_fasta.glob("*.fasta"))


def read_nanobody_sequence(fasta_path: Path) -> str:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    for rec in records:
        if rec.id == NANOBODY_CHAIN:
            return str(rec.seq)
    raise ValueError(f"No >{NANOBODY_CHAIN} record in {fasta_path}")


def read_fasta_sequence(fasta_path: Path) -> str:
    """Return nanobody sequence from a single-chain or >H FASTA."""
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No records in {fasta_path}")
    for rec in records:
        rid = rec.id.split(":")[-1] if ":" in rec.id else rec.id
        if rid == NANOBODY_CHAIN:
            return str(rec.seq)
    return str(records[0].seq)


def sequence_from_pdb(pdb_path: Path, chain_id: str | None = None) -> str:
    from Bio.SeqUtils import seq1

    structure = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    model = structure[0]
    chain = model[chain_id] if chain_id and chain_id in model else list(model.get_chains())[0]
    residues = [r for r in chain if r.id[0] == " "]
    return "".join(seq1(r.resname) for r in residues)


def extract_h_chain_pdb(src_pdb: Path, dst_pdb: Path, chain_id: str = NANOBODY_CHAIN) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("nanobody", str(src_pdb))
    model = structure[0]
    if chain_id not in [c.id for c in model]:
        raise ValueError(f"Chain {chain_id} not found in {src_pdb}")
    dst_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst_pdb), select=_ChainSelect(chain_id))


def write_aatype_fasta(sample_ids: list[str], dataset_dir: Path, out_fasta: Path) -> None:
    records = []
    native_dir = dataset_dir / "fasta.files.native"
    for sid in sample_ids:
        seq = read_nanobody_sequence(native_dir / f"{sid}.fasta")
        records.append(SeqRecord(Seq(seq), id=sid, description="nanobody_H"))
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(out_fasta), "fasta")


def extract_h_chain_pdbs(sample_ids: list[str], dataset_dir: Path, out_dir: Path) -> None:
    pdb_dir = dataset_dir / "pdb.native.files"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        src = pdb_dir / f"{sid}.pdb"
        dst = out_dir / f"{sid}.pdb"
        if not src.exists():
            raise FileNotFoundError(src)
        extract_h_chain_pdb(src, dst)


def tokenize_struct_fasta(
    pdb_dir: Path,
    out_dir: Path,
    struct_tokenizer_path: Path,
) -> Path:
    """Tokenize H-only PDBs with local struct tokenizer; return struct.fasta path."""
    from biotite.sequence.io import fasta as biotite_fasta
    from tqdm.auto import tqdm

    from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn
    from byprot.datamodules.pdb_dataset import utils as du
    from byprot.models.utils import get_struct_tokenizer
    from byprot.utils import recursive_to

    out_dir.mkdir(parents=True, exist_ok=True)
    struct_tokenizer = get_struct_tokenizer(str(struct_tokenizer_path))
    struct_tokenizer = struct_tokenizer.cuda().eval()

    all_data = []
    for pdb_path in sorted(pdb_dir.glob("*.pdb")):
        raw_chain_feats, metadata = du.process_pdb_file(str(pdb_path))
        chain_feats = struct_tokenizer.process_chain(raw_chain_feats)
        chain_feats["pdb_name"] = metadata["pdb_name"]
        chain_feats["pdb_path"] = str(pdb_path)
        chain_feats["header"] = pdb_path.stem
        all_data.append(chain_feats)

    dataloader = torch.utils.data.DataLoader(
        all_data, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    device = next(struct_tokenizer.parameters()).device
    header_struct_seq = []
    for batch in tqdm(dataloader, desc="tokenize_pdb"):
        header = batch["header"][0]
        batch = recursive_to(batch, device)
        struct_ids = struct_tokenizer.tokenize(
            batch["all_atom_positions"], batch["res_mask"], batch["seq_length"]
        )
        struct_seq = struct_tokenizer.struct_ids_to_seq(struct_ids.cpu().tolist()[0])
        header_struct_seq.append((header, struct_seq))

    struct_fasta = out_dir / "struct.fasta"
    biotite_fasta.FastaFile.write_iter(str(struct_fasta), header_struct_seq)
    return struct_fasta


def prepare_benchmark(
    dataset_dir: Path,
    benchmark_dir: Path,
    struct_tokenizer_path: Path | None = None,
    skip_tokenize: bool = False,
) -> dict[str, Path]:
    """Build aatype.fasta, pdb_h/, struct.fasta under benchmark_dir."""
    sample_ids = list_sample_ids(dataset_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    aatype_fasta = benchmark_dir / "aatype.fasta"
    pdb_h_dir = benchmark_dir / "pdb_h"
    tokenized_dir = benchmark_dir / "tokenized"

    write_aatype_fasta(sample_ids, dataset_dir, aatype_fasta)
    extract_h_chain_pdbs(sample_ids, dataset_dir, pdb_h_dir)
    cdr_regions_path = save_cdr_regions(dataset_dir, benchmark_dir)

    struct_fasta = benchmark_dir / "struct.fasta"
    if not skip_tokenize:
        if struct_tokenizer_path is None:
            raise ValueError("struct_tokenizer_path required unless --skip-tokenize")
        tokenize_struct_fasta(pdb_h_dir, tokenized_dir, struct_tokenizer_path)
        shutil.copy2(tokenized_dir / "struct.fasta", struct_fasta)
    elif not struct_fasta.exists():
        raise FileNotFoundError(
            f"{struct_fasta} missing. Re-run prepare with --struct-tokenizer."
        )

    manifest = {
        "dataset_dir": str(dataset_dir.resolve()),
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "aatype_fasta": str(aatype_fasta),
        "pdb_h_dir": str(pdb_h_dir),
    }
    if struct_fasta.exists():
        manifest["struct_fasta"] = str(struct_fasta)
    manifest["cdr_regions"] = str(cdr_regions_path)

    (benchmark_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    out = {
        "aatype_fasta": aatype_fasta,
        "pdb_h_dir": pdb_h_dir,
        "manifest": benchmark_dir / "manifest.json",
        "cdr_regions": cdr_regions_path,
    }
    if struct_fasta.exists():
        out["struct_fasta"] = struct_fasta
    return out


def list_igfold_nano_sample_ids(benchmark_root: Path) -> list[str]:
    igfold_dir = benchmark_root / "nano" / "IgFold"
    if not igfold_dir.is_dir():
        raise FileNotFoundError(f"Missing IgFold nanobody predictions: {igfold_dir}")
    return sorted(p.stem for p in igfold_dir.glob("*.pdb"))


def load_igfold_nano_stats(*stats_paths: Path) -> dict[str, int]:
    """Map PDB id -> CDR3 loop length (residues) from IgFold Nano/stats.csv."""
    out: dict[str, int] = {}
    for path in stats_paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            out[parts[0].strip()] = int(parts[1])
    return out


def build_chothia_cdr_regions(cdr3_len: int | None = None, max_resnum: int | None = None) -> dict:
    """CDR spans as inclusive Chothia PDB residue numbers (IgFold / ANARCI convention)."""
    cdr3_len = cdr3_len if cdr3_len is not None else CHOTHIA_VH_CDR3_DEFAULT_LEN
    cdr3_end_1 = CHOTHIA_VH_CDR3_START + cdr3_len - 1
    if max_resnum is not None:
        cdr3_end_1 = min(cdr3_end_1, max_resnum)
    return {
        "CDR1": CHOTHIA_VH_CDR1,
        "CDR2": CHOTHIA_VH_CDR2,
        "CDR3": (CHOTHIA_VH_CDR3_START, cdr3_end_1),
    }


def resolve_igfold_native_pdb(
    sample_id: str,
    xtal_dir: Path,
    native_extra_dir: Path | None = None,
) -> Path | None:
    candidates = [
        xtal_dir / "natives" / f"{sample_id}_trunc.pdb",
        xtal_dir / "natives" / f"{sample_id}.pdb",
    ]
    if native_extra_dir is not None:
        candidates.extend(
            [
                native_extra_dir / f"{sample_id}_trunc.pdb",
                native_extra_dir / f"{sample_id}.pdb",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_igfold_sequence(
    sample_id: str,
    xtal_dir: Path,
    igfold_pdb_dir: Path,
    native_extra_dir: Path | None = None,
) -> str:
    """Prefer native PDB sequence so folding input matches structural ground truth."""
    native_pdb = resolve_igfold_native_pdb(sample_id, xtal_dir, native_extra_dir)
    if native_pdb is not None:
        return sequence_from_pdb(native_pdb)
    fasta_path = xtal_dir / "fastas" / f"{sample_id}_trunc.fasta"
    if fasta_path.is_file():
        return read_fasta_sequence(fasta_path)
    pred_pdb = igfold_pdb_dir / f"{sample_id}.pdb"
    if pred_pdb.is_file():
        return sequence_from_pdb(pred_pdb)
    raise FileNotFoundError(
        f"No sequence for {sample_id}: missing native, {fasta_path}, and {pred_pdb}"
    )


def build_igfold_cdr_regions_map(
    sample_ids: list[str],
    sequences: dict[str, str],
    cdr3_lengths: dict[str, int],
    xtal_dir: Path,
    native_extra_dir: Path | None = None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sid in sample_ids:
        max_resnum = None
        native_pdb = resolve_igfold_native_pdb(sid, xtal_dir, native_extra_dir)
        if native_pdb is not None:
            feats = du.parse_pdb_feats(sid, str(native_pdb), chain_id=_first_chain_id(native_pdb))
            max_resnum = int(np.max(feats["residue_index"]))
        reg = build_chothia_cdr_regions(cdr3_lengths.get(sid), max_resnum=max_resnum)
        out[sid] = {
            "length": len(sequences[sid]),
            "cdr1": list(reg["CDR1"]),
            "cdr2": list(reg["CDR2"]),
            "cdr3": list(reg["CDR3"]),
        }
    return out


def prepare_igfold_nano_benchmark(
    benchmark_root: Path,
    benchmark_dir: Path,
    xtal_dir: Path | None = None,
    native_extra_dir: Path | None = None,
    struct_tokenizer_path: Path | None = None,
    skip_tokenize: bool = False,
) -> dict[str, Path]:
    """Prepare IgFold nanobody benchmark (71 targets under benchmark/nano/IgFold)."""
    xtal_dir = xtal_dir or DEFAULT_IGFOLD_XTAL_DIR
    igfold_pdb_dir = benchmark_root / "nano" / "IgFold"
    sample_ids = list_igfold_nano_sample_ids(benchmark_root)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    stats_paths = [
        Path.home() / "IgFold_benchmark" / "IgFold" / "Nano" / "stats.csv",
        benchmark_root / "nano" / "stats.csv",
    ]
    cdr3_lengths = load_igfold_nano_stats(*stats_paths)

    sequences: dict[str, str] = {}
    native_sources: dict[str, str] = {}
    for sid in sample_ids:
        native_pdb = resolve_igfold_native_pdb(sid, xtal_dir, native_extra_dir)
        sequences[sid] = resolve_igfold_sequence(
            sid, xtal_dir, igfold_pdb_dir, native_extra_dir
        )
        if native_pdb is not None:
            native_sources[sid] = str(native_pdb)

    aatype_fasta = benchmark_dir / "aatype.fasta"
    pdb_h_dir = benchmark_dir / "pdb_h"
    pdb_h_dir.mkdir(parents=True, exist_ok=True)
    records = [SeqRecord(Seq(sequences[sid]), id=sid, description="igfold_nano") for sid in sample_ids]
    SeqIO.write(records, str(aatype_fasta), "fasta")

    for sid, src in native_sources.items():
        shutil.copy2(src, pdb_h_dir / f"{sid}.pdb")

    cdr_regions_path = benchmark_dir / "cdr_regions.json"
    cdr_regions_path.write_text(
        json.dumps(
            build_igfold_cdr_regions_map(
                sample_ids, sequences, cdr3_lengths, xtal_dir, native_extra_dir
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    struct_fasta = benchmark_dir / "struct.fasta"
    tokenized_dir = benchmark_dir / "tokenized"
    if not skip_tokenize:
        if struct_tokenizer_path is None:
            raise ValueError("struct_tokenizer_path required unless --skip-tokenize")
        if not any(pdb_h_dir.glob("*.pdb")):
            raise FileNotFoundError(
                f"No native PDBs in {pdb_h_dir}. Provide crystal structures via "
                f"--xtal-dir or --native-dir (only {len(native_sources)}/{len(sample_ids)} found)."
            )
        tokenize_struct_fasta(pdb_h_dir, tokenized_dir, struct_tokenizer_path)
        shutil.copy2(tokenized_dir / "struct.fasta", struct_fasta)
    elif not skip_tokenize and not struct_fasta.exists():
        raise FileNotFoundError(
            f"{struct_fasta} missing. Re-run prepare with --struct-tokenizer."
        )

    manifest = {
        "benchmark": "igfold_nano",
        "benchmark_root": str(benchmark_root.resolve()),
        "xtal_dir": str(xtal_dir.resolve()),
        "n_samples": len(sample_ids),
        "n_with_native": len(native_sources),
        "sample_ids": sample_ids,
        "sample_ids_with_native": sorted(native_sources.keys()),
        "aatype_fasta": str(aatype_fasta),
        "pdb_h_dir": str(pdb_h_dir),
        "cdr_regions": str(cdr_regions_path),
        "cdr_numbering": "chothia_pdb",
    }
    if struct_fasta.exists():
        manifest["struct_fasta"] = str(struct_fasta)
    (benchmark_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out: dict[str, Path] = {
        "aatype_fasta": aatype_fasta,
        "pdb_h_dir": pdb_h_dir,
        "manifest": benchmark_dir / "manifest.json",
        "cdr_regions": cdr_regions_path,
    }
    if struct_fasta.exists():
        out["struct_fasta"] = struct_fasta
    return out


def load_native_sequences(benchmark_dir: Path) -> dict[str, str]:
    seqs = {}
    for rec in SeqIO.parse(str(benchmark_dir / "aatype.fasta"), "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs


def _read_h_chain_fasta(fasta_path: Path) -> str:
    return read_nanobody_sequence(fasta_path)


def infer_cdr_range(native_seq: str, design_seq: str) -> tuple[int, int] | None:
    """Return inclusive 0-based [start, end] where design masks CDR with X."""
    if len(native_seq) != len(design_seq):
        return None
    idx = [i for i, (a, b) in enumerate(zip(native_seq, design_seq)) if a != b]
    if not idx:
        return None
    return min(idx), max(idx)


def infer_cdr_regions_for_sample(dataset_dir: Path, sample_id: str) -> dict:
    """CDR1/2/3 ranges from fasta.files.design/h_cdr{1,2,3} masks (X vs native)."""
    native_path = dataset_dir / "fasta.files.native" / f"{sample_id}.fasta"
    native_seq = _read_h_chain_fasta(native_path)
    regions: dict = {"length": len(native_seq)}
    for cdr_num in (1, 2, 3):
        design_path = dataset_dir / "fasta.files.design" / f"h_cdr{cdr_num}" / f"{sample_id}.fasta"
        if not design_path.exists():
            raise FileNotFoundError(design_path)
        design_seq = _read_h_chain_fasta(design_path)
        span = infer_cdr_range(native_seq, design_seq)
        if span is None:
            raise ValueError(f"Could not infer CDR{cdr_num} for {sample_id}")
        regions[f"CDR{cdr_num}"] = span
    return regions


def build_cdr_regions_map(dataset_dir: Path, sample_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sid in sample_ids:
        reg = infer_cdr_regions_for_sample(dataset_dir, sid)
        out[sid] = {
            "length": reg["length"],
            "cdr1": list(reg["CDR1"]),
            "cdr2": list(reg["CDR2"]),
            "cdr3": list(reg["CDR3"]),
        }
    return out


def save_cdr_regions(dataset_dir: Path, benchmark_dir: Path) -> Path:
    sample_ids = list_sample_ids(dataset_dir)
    regions = build_cdr_regions_map(dataset_dir, sample_ids)
    path = benchmark_dir / "cdr_regions.json"
    path.write_text(json.dumps(regions, indent=2), encoding="utf-8")
    return path


def load_cdr_numbering(benchmark_dir: Path) -> str:
    manifest_path = benchmark_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "cdr_numbering", "sequence_0based"
        )
    return "sequence_0based"


def load_cdr_regions(benchmark_dir: Path) -> dict[str, dict]:
    path = benchmark_dir / "cdr_regions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Re-run prepare_sab23_h2_nano_benchmark.py on the dataset."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[str, dict] = {}
    for sid, reg in raw.items():
        parsed[sid] = {
            "CDR1": tuple(reg["cdr1"]),
            "CDR2": tuple(reg["cdr2"]),
            "CDR3": tuple(reg["cdr3"]),
            "length": reg["length"],
        }
    return parsed


def _array_indices_for_chothia_span(
    residue_index: np.ndarray,
    start_1: int,
    end_1: int,
) -> np.ndarray:
    """Map inclusive Chothia residue numbers to PDB array indices."""
    return np.where((residue_index >= start_1) & (residue_index <= end_1))[0]


def _framework_indices(n: int, regions: dict) -> np.ndarray:
    cdr = np.zeros(n, dtype=bool)
    for key in ("CDR1", "CDR2", "CDR3"):
        start, end = regions[key]
        cdr[start : end + 1] = True
    return np.where(~cdr)[0]


def _clip_span(start: int, end: int, n: int) -> tuple[int, int] | None:
    """Clip inclusive CDR span to sequence length n; return None if entirely out of range."""
    if start >= n:
        return None
    return start, min(end, n - 1)


def _indices_for_region(
    regions: dict,
    region: str,
    n: int,
    residue_index: np.ndarray | None = None,
) -> np.ndarray | None:
    if residue_index is not None:
        return _indices_for_region_chothia(regions, region, residue_index)
    if region == "Fr":
        return _framework_indices(n, regions)
    if region not in regions:
        raise KeyError(region)
    clipped = _clip_span(regions[region][0], regions[region][1], n)
    if clipped is None:
        return None
    start, end = clipped
    return np.arange(start, end + 1)


def _indices_for_region_chothia(
    regions: dict,
    region: str,
    residue_index: np.ndarray,
) -> np.ndarray | None:
    """Region masks using Chothia residue numbers stored in regions (1-based)."""
    if region == "Fr":
        cdr_mask = np.zeros(len(residue_index), dtype=bool)
        for key in ("CDR1", "CDR2", "CDR3"):
            start, end = regions[key]
            cdr_mask |= (residue_index >= start) & (residue_index <= end)
        idx = np.where(~cdr_mask)[0]
        return idx if len(idx) else None
    if region not in regions:
        raise KeyError(region)
    start, end = regions[region]
    idx = _array_indices_for_chothia_span(residue_index, start, end)
    return idx if len(idx) else None


def _align_pred_onto_gt(
    pred_ca: np.ndarray,
    gt_ca: np.ndarray,
    align_indices: np.ndarray,
) -> np.ndarray:
    """Rigid-body align all pred CA onto gt using only align_indices for the fit."""
    ref = gt_ca[align_indices]
    mobile = pred_ca[align_indices]
    sup = SVDSuperimposer()
    sup.set(ref, mobile)
    sup.run()
    rot, tran = sup.get_rotran()
    return np.dot(pred_ca, rot) + tran


def _ca_rmsd_after_align(
    pred_ca: np.ndarray,
    gt_ca: np.ndarray,
    eval_indices: np.ndarray,
    align_indices: np.ndarray | None = None,
) -> float:
    """Superimpose on align_indices (default: all CA), then RMSD on eval_indices."""
    n = min(len(pred_ca), len(gt_ca))
    pred_ca = pred_ca[:n]
    gt_ca = gt_ca[:n]
    if align_indices is None or len(align_indices) == 0:
        align_indices = np.arange(n)
    else:
        align_indices = align_indices[(align_indices >= 0) & (align_indices < n)]
    eval_indices = eval_indices[(eval_indices >= 0) & (eval_indices < n)]
    if len(align_indices) == 0 or len(eval_indices) == 0:
        return float("nan")

    if len(align_indices) == n:
        align_mask = torch.ones(n, dtype=torch.bool)
        aligned_pred, _ = superimpose(
            torch.tensor(gt_ca, dtype=torch.float32)[None],
            torch.tensor(pred_ca, dtype=torch.float32)[None],
            align_mask,
        )
        aligned_pred = aligned_pred[0].numpy()
    else:
        aligned_pred = _align_pred_onto_gt(pred_ca, gt_ca, align_indices)
    diff = aligned_pred[eval_indices] - gt_ca[eval_indices]
    return float(np.sqrt((diff**2).sum(axis=-1).mean()))


def _seq_recovery_range(pred: str, native: str, start: int, end: int) -> float:
    pred_sub = pred[start : end + 1]
    native_sub = native[start : end + 1]
    if not native_sub:
        return float("nan")
    return sum(a == b for a, b in zip(pred_sub, native_sub)) / len(native_sub)


def _first_chain_id(pdb_path: Path) -> str:
    structure = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    chains = list(structure[0].get_chains())
    if not chains:
        raise ValueError(f"No chains in {pdb_path}")
    return chains[0].id


def _chain_feats(pdb_path: Path) -> dict:
    chain_id = _first_chain_id(pdb_path)
    return du.parse_pdb_feats(pdb_path.stem, str(pdb_path), chain_id=chain_id)


def _sequence_pair_indices(pred_seq: str, gt_seq: str) -> tuple[np.ndarray, np.ndarray]:
    """Map pred/gt PDB row indices for identical aligned residues (global alignment)."""
    if pred_seq == gt_seq:
        idx = np.arange(len(pred_seq))
        return idx, idx.copy()
    # Common IgFold case: xtal FASTA has one extra N-terminal residue vs native PDB.
    if len(pred_seq) == len(gt_seq) + 1 and pred_seq[1:] == gt_seq:
        return np.arange(1, len(pred_seq)), np.arange(len(gt_seq))
    if len(gt_seq) == len(pred_seq) + 1 and gt_seq[1:] == pred_seq:
        return np.arange(len(pred_seq)), np.arange(1, len(gt_seq))

    from Bio import pairwise2

    aln = pairwise2.align.globalxx(pred_seq, gt_seq, one_alignment_only=True)[0]
    pred_idx: list[int] = []
    gt_idx: list[int] = []
    pi = gi = 0
    for a, b in zip(aln[0], aln[1]):
        if a != "-" and b != "-":
            pred_idx.append(pi)
            gt_idx.append(gi)
        if a != "-":
            pi += 1
        if b != "-":
            gi += 1
    return np.asarray(pred_idx, dtype=int), np.asarray(gt_idx, dtype=int)


def _pair_structure_to_reference(pred_feats: dict, gt_feats: dict) -> dict | None:
    pred_seq = du.aatype_to_seq(pred_feats["aatype"])
    gt_seq = du.aatype_to_seq(gt_feats["aatype"])
    pred_idx, gt_idx = _sequence_pair_indices(pred_seq, gt_seq)
    if len(pred_idx) < 3:
        return None
    identity = sum(pred_seq[i] == gt_seq[j] for i, j in zip(pred_idx, gt_idx)) / len(pred_idx)
    return {
        "pred_ca": np.asarray(pred_feats["bb_positions"])[pred_idx],
        "gt_ca": np.asarray(gt_feats["bb_positions"])[gt_idx],
        "pred_bb_n3": pred_feats["atom_positions"][pred_idx, :3],
        "gt_bb_n3": gt_feats["atom_positions"][gt_idx, :3],
        "pred_seq": "".join(pred_seq[i] for i in pred_idx),
        "gt_seq": "".join(gt_seq[i] for i in gt_idx),
        "gt_idx": gt_idx,
        "gt_residue_index": np.asarray(gt_feats["residue_index"])[gt_idx],
        "n_aligned": int(len(pred_idx)),
        "seq_identity": float(identity),
    }


def _region_indices_matched(
    regions: dict,
    region: str,
    gt_idx: np.ndarray,
    gt_residue_index: np.ndarray | None,
    cdr_numbering: str,
) -> np.ndarray | None:
    """Region mask on sequence-aligned (matched) residue pairs."""
    n = len(gt_idx)
    use_chothia = cdr_numbering in ("chothia", "chothia_pdb")
    if use_chothia and gt_residue_index is not None:
        return _indices_for_region_chothia(regions, region, gt_residue_index)
    if region == "Fr":
        cdr_mask = np.zeros(n, dtype=bool)
        for key in ("CDR1", "CDR2", "CDR3"):
            start, end = regions[key]
            cdr_mask |= (gt_idx >= start) & (gt_idx <= end)
        idx = np.where(~cdr_mask)[0]
        return idx if len(idx) else None
    if region not in regions:
        raise KeyError(region)
    start, end = regions[region]
    idx = np.where((gt_idx >= start) & (gt_idx <= end))[0]
    return idx if len(idx) else None


def compute_folding_metrics(
    pred_pdb_dir: Path,
    gt_pdb_dir: Path,
    cdr_regions: dict[str, dict] | None = None,
    cdr_numbering: str = "sequence_0based",
) -> list[dict]:
    rows = []
    pred_pdbs = sorted(pred_pdb_dir.glob("*.pdb"))
    if not pred_pdbs:
        return rows
    for pred_pdb in pred_pdbs:
        sid = pred_pdb.stem
        gt_pdb = gt_pdb_dir / f"{sid}.pdb"
        row = {"sample_id": sid}
        if not gt_pdb.exists():
            row.update({"status": "missing_gt", "bb_rmsd": None, "ca_rmsd": None, "bb_tmscore": None})
            rows.append(row)
            continue
        try:
            pred = _chain_feats(pred_pdb)
            gt = _chain_feats(gt_pdb)
            paired = _pair_structure_to_reference(pred, gt)
            if paired is None:
                row.update({"status": "align_failed", "bb_rmsd": None, "ca_rmsd": None, "bb_tmscore": None})
                rows.append(row)
                continue
            if paired["seq_identity"] < 0.95:
                row.update(
                    {
                        "status": f"low_seq_identity:{paired['seq_identity']:.2f}",
                        "bb_rmsd": None,
                        "ca_rmsd": None,
                        "bb_tmscore": None,
                    }
                )
                rows.append(row)
                continue

            pred_ca = paired["pred_ca"]
            gt_ca = paired["gt_ca"]
            pred_bb_n3 = paired["pred_bb_n3"]
            gt_bb_n3 = paired["gt_bb_n3"]
            n = paired["n_aligned"]
            mask = torch.ones(n, dtype=torch.bool)

            ca_rmsd = superimpose(
                torch.tensor(pred_ca)[None],
                torch.tensor(gt_ca)[None],
                mask,
            )[1].item()
            pred_bb_flat = pred_bb_n3.reshape(-1, 3)
            gt_bb_flat = gt_bb_n3.reshape(-1, 3)
            bb_rmsd = superimpose(
                torch.tensor(pred_bb_flat)[None],
                torch.tensor(gt_bb_flat)[None],
                mask[:, None].repeat(1, 3).reshape(-1),
            )[1].item()
            _, tmscore = calc_tm_score(
                pred_bb_n3, gt_bb_n3, paired["pred_seq"], paired["gt_seq"]
            )
            row.update(
                {
                    "status": "ok",
                    "bb_rmsd": float(bb_rmsd),
                    "ca_rmsd": float(ca_rmsd),
                    "bb_tmscore": float(tmscore),
                    "length": n,
                    "n_aligned": n,
                    "seq_identity": paired["seq_identity"],
                }
            )
            if cdr_regions and sid in cdr_regions:
                reg = cdr_regions[sid]
                gt_residx = (
                    paired["gt_residue_index"]
                    if cdr_numbering in ("chothia", "chothia_pdb")
                    else None
                )
                fr_idx = _region_indices_matched(
                    reg, "Fr", paired["gt_idx"], gt_residx, cdr_numbering
                )
                align_idx = fr_idx if fr_idx is not None and len(fr_idx) >= 3 else np.arange(n)
                for region in REGION_NAMES:
                    idx = _region_indices_matched(
                        reg, region, paired["gt_idx"], gt_residx, cdr_numbering
                    )
                    if idx is None or len(idx) == 0:
                        row[f"{region}_ca_rmsd"] = float("nan")
                    else:
                        row[f"{region}_ca_rmsd"] = _ca_rmsd_after_align(
                            pred_ca, gt_ca, idx, align_indices=align_idx
                        )
        except Exception as exc:
            row.update({"status": f"error:{exc}", "bb_rmsd": None, "ca_rmsd": None, "bb_tmscore": None})
        rows.append(row)
    return rows


def compute_inverse_folding_metrics(
    pred_aatype_fasta: Path,
    native_seqs: dict[str, str],
    cdr_regions: dict[str, dict] | None = None,
) -> list[dict]:
    rows = []
    for rec in SeqIO.parse(str(pred_aatype_fasta), "fasta"):
        sid = rec.id
        pred = str(rec.seq)
        native = native_seqs.get(sid)
        row = {"sample_id": sid}
        if native is None:
            row.update({"status": "missing_native", "seq_recovery": None})
        else:
            n = min(len(pred), len(native))
            if n == 0:
                recovery = 0.0
            else:
                recovery = sum(pred[i] == native[i] for i in range(n)) / n
            row.update(
                {
                    "status": "ok",
                    "seq_recovery": float(recovery),
                    "length": n,
                    "length_match": len(pred) == len(native),
                }
            )
            if cdr_regions and sid in cdr_regions:
                reg = cdr_regions[sid]
                for region in REGION_NAMES:
                    if region == "Fr":
                        idx = _indices_for_region(reg, "Fr", n)
                        row["Fr_seq_recovery"] = (
                            float(np.mean([pred[i] == native[i] for i in idx]))
                            if idx is not None and len(idx)
                            else float("nan")
                        )
                    else:
                        clipped = _clip_span(reg[region][0], reg[region][1], n)
                        if clipped is None:
                            row[f"{region}_seq_recovery"] = float("nan")
                        else:
                            row[f"{region}_seq_recovery"] = _seq_recovery_range(
                                pred, native, clipped[0], clipped[1]
                            )
        rows.append(row)
    return rows


def summarize_region_metrics(rows: list[dict], metric_suffix: str) -> dict[str, dict]:
    """metric_suffix: 'ca_rmsd' (folding) or 'seq_recovery' (inverse folding)."""
    out: dict[str, dict] = {}
    for region in REGION_NAMES:
        col = f"{region}_{metric_suffix}"
        summary = summarize_metrics(rows, col)
        if summary.get("count", 0) > 0:
            out[region] = summary
    return out


def print_region_table(title: str, region_summary: dict[str, dict]) -> None:
    print(f"\n{title}")
    header = "  ".join(f"{r:>8}" for r in REGION_NAMES)
    print(f"  {'':8} {header}")
    for stat in ("mean", "median"):
        values = []
        for region in REGION_NAMES:
            val = region_summary.get(region, {}).get(stat)
            values.append(f"{val:>8.3f}" if val is not None else f"{'n/a':>8}")
        print(f"  {stat:8} {'  '.join(values)}")


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def summarize_metrics(rows: list[dict], value_key: str) -> dict:
    vals = [
        r[value_key]
        for r in rows
        if r.get(value_key) is not None and not _is_nan(r[value_key])
    ]
    if not vals:
        return {"count": 0}
    arr = np.array(vals, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
