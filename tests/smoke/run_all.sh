#!/usr/bin/env bash
# Run all DPLM-2 smoke tests from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SMOKE_DIR="$REPO_ROOT/tests/smoke"
OUT_DIR="${SMOKE_OUT_DIR:-$REPO_ROOT/tests/smoke_outputs}"
MODEL="${DPLM2_MODEL:-$REPO_ROOT/checkpoints/dplm2_150m}"
STOK="${DPLM2_STRUCT_TOKENIZER:-$REPO_ROOT/checkpoints/struct_tokenizer}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "== DPLM-2 smoke tests =="
echo "repo:   $REPO_ROOT"
echo "model:  $MODEL"
echo "stok:   $STOK"
echo "output: $OUT_DIR"
echo

COMMON_ARGS=(--model "$MODEL" --struct-tokenizer "$STOK")

run() {
  echo ">> $*"
  python "$@"
  echo
}

run "$SMOKE_DIR/01_load_model.py" "${COMMON_ARGS[@]}"

run "$SMOKE_DIR/02_co_generation.py" "${COMMON_ARGS[@]}" \
  --output "$OUT_DIR/co_generation" \
  --length 50 --num-seqs 1 --max-iter 50

if [[ -f "$REPO_ROOT/data-bin/cameo2022_mini/aatype.fasta" ]]; then
  run "$SMOKE_DIR/03_folding_benchmark.py" "${COMMON_ARGS[@]}" \
    --output "$OUT_DIR/folding" --max-iter 50
  run "$SMOKE_DIR/04_inverse_folding_benchmark.py" "${COMMON_ARGS[@]}" \
    --output "$OUT_DIR/inverse_folding" --max-iter 50
else
  echo ">> benchmark tests skipped (no data-bin/cameo2022_mini)"
  echo "   Run: bash scripts/download_benchmark_data.sh"
  echo "        python tests/smoke/prepare_benchmark_subset.py"
  echo
fi

echo "== All smoke tests finished =="
