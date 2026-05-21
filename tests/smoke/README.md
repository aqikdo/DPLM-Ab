# DPLM-2 Smoke Tests

Quick checks for environment, model loading, generation, and optional benchmark subsets.

## Prerequisites

```bash
conda activate dplm
# torch_scatter (if import fails):
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# struct tokenizer / PDB export:
pip install fairscale

# Model weights (150M example)
huggingface-cli download airkingbd/dplm2_150m --local-dir checkpoints/dplm2_150m
huggingface-cli download airkingbd/struct_tokenizer --local-dir checkpoints/struct_tokenizer_download
mkdir -p checkpoints/struct_tokenizer/.hydra
mv checkpoints/struct_tokenizer_download/{config.yaml,dplm2_struct_tokenizer.ckpt} \
   checkpoints/struct_tokenizer/.hydra/
```

## Run all tests

From repo root:

```bash
bash tests/smoke/run_all.sh
```

Or individually:

```bash
cd /path/to/dplm
export PYTHONPATH=.

python tests/smoke/01_load_model.py
python tests/smoke/02_co_generation.py --length 50 --max-iter 50
```

## Benchmark data (optional)

README benchmarks use Zenodo metadata ([`scripts/download_metadata.sh`](../scripts/download_metadata.sh)):

| Benchmark | Path after download | Smoke test |
|-----------|---------------------|------------|
| CAMEO 2022 folding | `data-bin/cameo2022/aatype.fasta` | `03_folding_benchmark.py` |
| CAMEO 2022 inverse folding | `data-bin/cameo2022/struct.fasta` | `04_inverse_folding_benchmark.py` |
| PDB date split | `data-bin/PDB_date/` | (use same scripts with `--input-fasta`) |
| Motif scaffolding | `data-bin/scaffolding-pdbs/` | `run/scaffold_generate_dplm2.py` (full pipeline) |
| Co-generation metrics | generated FASTA dir | `evaluator_dplm2.py -cn unconditional_codesign` |

One-shot download + mini subset:

```bash
bash scripts/download_benchmark_data.sh
python tests/smoke/03_folding_benchmark.py
python tests/smoke/04_inverse_folding_benchmark.py
```

Mini subset (`data-bin/cameo2022_mini/`, 3 proteins) keeps smoke runs fast. Full paper metrics need full CAMEO + `evaluator_dplm2.py` with `max_iter=100`.

### Metrics evaluation (slow)

```bash
python tests/smoke/03_folding_benchmark.py --run-eval
# Requires data-bin/metadata/pdb_afdb_cameo.csv and ground-truth structures
```

## Environment variables

| Variable | Default |
|----------|---------|
| `DPLM2_MODEL` | `checkpoints/dplm2_150m` |
| `DPLM2_STRUCT_TOKENIZER` | `checkpoints/struct_tokenizer` |
| `SMOKE_OUT_DIR` | `tests/smoke_outputs` |

## Outputs

Results go to `tests/smoke_outputs/` (gitignored): FASTA, PDB, optional eval CSVs.

## Test list

| Script | What it checks |
|--------|----------------|
| `01_load_model.py` | Load weights + struct tokenizer |
| `02_co_generation.py` | Short unconditional co-generation + PDB |
| `03_folding_benchmark.py` | Forward folding on CAMEO mini |
| `04_inverse_folding_benchmark.py` | Inverse folding on CAMEO mini |
| `prepare_benchmark_subset.py` | Build `cameo2022_mini` from full CAMEO |

Default smoke settings: `length=50`, `max_iter=50`, `num_seqs=1` (minutes on one GPU). Paper settings: `max_iter=500`, full benchmark.
