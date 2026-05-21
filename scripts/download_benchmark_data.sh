#!/usr/bin/env bash
# Download DPLM-2 evaluation benchmark data (CAMEO 2022, PDB date, motif scaffolds, metadata).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Downloading dplm2_metadata from Zenodo (~check size before download)..."
bash scripts/download_metadata.sh

echo
echo "Creating mini CAMEO subset for smoke tests (3 proteins)..."
python tests/smoke/prepare_benchmark_subset.py --max-proteins 3

echo
echo "Benchmark data ready under data-bin/"
echo "  cameo2022/       - full CAMEO 2022 (folding / inverse folding)"
echo "  cameo2022_mini/  - 3-protein subset for fast smoke tests"
echo "  metadata/        - CSV for evaluator metrics"
