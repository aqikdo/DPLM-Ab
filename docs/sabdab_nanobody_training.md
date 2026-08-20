# SAbDab strict nanobody (VHH) DPLM-2 training

This pipeline turns a local [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab/) mirror into DPLM-2 training tokens (`aa_seq` + comma-separated `struct_seq`) and runs LoRA finetuning on the official 650M checkpoint.

## Inclusion rules (strict nanobody)

| Condition | Action |
|-----------|--------|
| FASTA contains `:L` (light chain) | **Exclude** |
| PDB contains chain `L` | **Exclude** (even if FASTA omits L) |
| FASTA has only `>id:H` or a single non-L chain | **Include**; extract that chain from PDB |
| PDB has H + antigen chains | **Include**; only the antibody chain is tokenized |
| FASTA sequence ≠ PDB chain sequence | **Exclude** (logged in `rejected.jsonl`) |
| Length outside `[80, 512]` (default) | **Exclude** |
| PDB id in `--exclude-ids-file` | **Exclude** (e.g. IgFold benchmark overlap) |

**Note:** Validation split uses **cluster = pdb_id** with random 5% of clusters held out (no resolution/date metadata in a plain `~/sabdab` folder). Optional MMseqs2 clustering can be added later by updating the `cluster` column before `save_to_disk`.

## Prerequisites

- `~/sabdab` with paired `{pdbid}.pdb` and `{pdbid}.fasta`
- Conda env `dplm` with repo dependencies
- Struct tokenizer under `checkpoints/struct_tokenizer` (or pass `--struct-tokenizer`)
- DPLM-2 base weights (downloaded automatically via Hugging Face on first train)

## 1. Prepare dataset

```bash
cd /home/zhaohongyan/dplm
conda activate dplm
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH

# Quick filter-only smoke (no GPU)
python scripts/prepare_sabdab_nanobody_for_dplm2.py \
  --sabdab-dir ~/sabdab \
  --output-dir data-bin/sabdab_nanobody \
  --filter-only \
  --limit 100

# Full prepare (GPU recommended; ~647 strict VHH entries expected)
python scripts/prepare_sabdab_nanobody_for_dplm2.py \
  --sabdab-dir ~/sabdab \
  --output-dir data-bin/sabdab_nanobody \
  --struct-tokenizer checkpoints/struct_tokenizer \
  --val-ratio 0.05

# Optional: exclude IgFold benchmark PDB ids
python scripts/prepare_sabdab_nanobody_for_dplm2.py \
  ... \
  --exclude-ids-file data-bin/igfold_nano/manifest.json
```

Output layout:

```
data-bin/sabdab_nanobody/
  metadata.csv
  rejected.jsonl
  manifest.json
  train/          # HuggingFace dataset on disk
  valid/
```

## 2. Train (LoRA)

```bash
bash scripts/train_sabdab_nanobody.sh

# Smoke (single GPU, few steps; uses data-bin/sabdab_nanobody_smoke after prepare --tokenize-limit 8)
python train.py \
  experiment=dplm2/dplm2_650m_sabdab_nanobody \
  name=dplm2_650m_sabdab_nanobody_smoke \
  datamodule.csv_file=sabdab_nanobody_smoke \
  trainer.max_steps=10 \
  trainer.devices=1 \
  trainer.enable_progress_bar=false \
  datamodule.num_workers=0 \
  datamodule.max_tokens=4096

Default config loads local **`checkpoints/dplm2_650m`** (multimodal DPLM-2, `continue_train_from_dplm2`). No need to download `dplm_650m` unless you switch back to `training_stage: train_from_dplm`.
```

Hydra experiment: `configs/experiment/dplm2/dplm2_650m_sabdab_nanobody.yaml` (inherits `dplm2_650m`, sets `datamodule.csv_file: sabdab_nanobody`).

## 3. Optional benchmark after finetune

```bash
python tests/smoke/07_sab23_h2_nano_benchmark.py \
  --benchmark igfold \
  --checkpoint-dir logs/dplm2_650m_sabdab_nanobody/checkpoints
```

## Troubleshooting

| Issue | Mitigation |
|-------|------------|
| OOM during train | Lower `datamodule.max_tokens` or use `trainer.accumulate_grad_batches` |
| Prepare slow | Use `--limit` for tests; `--resume` reuses `struct_seq` in `metadata.csv` |
| Very few included samples | ~647 FASTA without L; ~416 pass PDB sequence check after filters |
| FASTA missing L but PDB has L | PDB `L` check rejects these |

## Known limitations

- No antigen-conditioned training (see repo `plan.md` for future scFv/adapter work).
- No automatic SAbDab download.
- Distribution shift vs `pdb_swissprot`; monitor `val/loss` when finetuning.
