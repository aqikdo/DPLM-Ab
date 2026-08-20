#!/usr/bin/env bash
# Sequential nano43 invfold baseline runner (protocol A).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PROJECT_ROOT="$PWD"
export PYTHONPATH="${PWD}/src:${PWD}:${PWD}/scripts:${PWD}/tests/smoke:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PYTHON="${PYTHON:-/home/zhaohongyan/miniconda3/envs/dplm/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT_ROOT="outputs/vhh_eval_ext_invfold_baselines"
SAMPLES="${OUT_ROOT}/_inputs/samples.json"
PMPNN_DIR="third_party/ProteinMPNN"
mkdir -p "$OUT_ROOT" logs
SKIPPED="${OUT_ROOT}/SKIPPED.md"
: > "$SKIPPED"
echo "# Skipped methods" >> "$SKIPPED"
echo >> "$SKIPPED"
echo "- IgMPNN/IgDesign: weights/training data not fully public." >> "$SKIPPED"

score_one() {
  local tag="$1" ag="$2" notes="$3"
  local dir="${OUT_ROOT}/${tag}"
  local extra=()
  [[ "$ag" == "1" ]] && extra+=(--ag-aware)
  "${PYTHON}" scripts/baselines/score_nano43_invfold.py \
    --pred-fasta "${dir}/predictions.fasta" \
    --out-dir "${dir}" \
    --tag "${tag}" \
    --notes "${notes}" \
    "${extra[@]}"
  "${PYTHON}" scripts/baselines/update_comparison.py --out-root "${OUT_ROOT}"
}

wait_file_size() {
  # wait until file exists and size stable for 10s and >= min_bytes
  local path="$1" min_bytes="$2" max_wait="${3:-7200}"
  local t=0 last=-1
  while (( t < max_wait )); do
    if [[ -f "$path" ]]; then
      local sz
      sz=$(stat -c%s "$path")
      if (( sz >= min_bytes )) && (( sz == last )); then
        sleep 5
        sz=$(stat -c%s "$path")
        if (( sz == last )); then
          echo "ready $path size=$sz"
          return 0
        fi
      fi
      last=$sz
    fi
    sleep 10
    t=$((t+10))
    echo "[wait ${t}s] $path size=${last}"
  done
  return 1
}

echo "[$(date)] prepare inputs"
"${PYTHON}" scripts/baselines/prepare_nano43_invfold_inputs.py

# --- 1 ProteinMPNN ---
echo "[$(date)] ProteinMPNN weights"
# IPD file may be named proteinmpnn_*; we store as v_48_020.pt
PMPNN_W="third_party/pmpnn_weights/v_48_020.pt"
# expected ~6.7MB; accept >= 6_000_000
if ! wait_file_size "$PMPNN_W" 6000000 7200; then
  echo "- ProteinMPNN: weight download incomplete ($PMPNN_W)" >> "$SKIPPED"
else
  # validate torch load
  if "${PYTHON}" -c "import torch; torch.load('${PMPNN_W}', map_location='cpu'); print('ok')"; then
    mkdir -p "${PMPNN_DIR}/vanilla_model_weights"
    cp -f "$PMPNN_W" "${PMPNN_DIR}/vanilla_model_weights/v_48_020.pt"
    DIR="${OUT_ROOT}/ProteinMPNN"
    mkdir -p "$DIR"
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON}" -u scripts/baselines/run_proteinmpnn_nano43.py \
      --samples-json "$SAMPLES" \
      --pmpnn-dir "$PMPNN_DIR" \
      --out-dir "$DIR" \
      --path-to-model-weights "${PMPNN_DIR}/vanilla_model_weights" \
      --model-name v_48_020 \
      > "logs/ext_invfold_ProteinMPNN.log" 2>&1
    score_one ProteinMPNN 1 "vanilla ProteinMPNN; design H, fix Ag"
  else
    echo "- ProteinMPNN: weight corrupt/incomplete" >> "$SKIPPED"
  fi
fi

# --- 2 AbMPNN ---
echo "[$(date)] AbMPNN"
ABW="third_party/abmpnn_weights/abmpnn.pt"
if ! wait_file_size "$ABW" 19000000 7200; then
  echo "- AbMPNN: weight download incomplete" >> "$SKIPPED"
else
  if "${PYTHON}" -c "import torch; d=torch.load('${ABW}', map_location='cpu'); print(list(d.keys())[:8])"; then
    mkdir -p "${PMPNN_DIR}/vanilla_model_weights"
    # ProteinMPNN loads {model_name}.pt expecting keys model_state_dict/num_edges/noise_level
    cp -f "$ABW" "${PMPNN_DIR}/vanilla_model_weights/abmpnn.pt"
    DIR="${OUT_ROOT}/AbMPNN"
    mkdir -p "$DIR"
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON}" -u scripts/baselines/run_proteinmpnn_nano43.py \
      --samples-json "$SAMPLES" \
      --pmpnn-dir "$PMPNN_DIR" \
      --out-dir "$DIR" \
      --path-to-model-weights "${PMPNN_DIR}/vanilla_model_weights" \
      --model-name abmpnn \
      > "logs/ext_invfold_AbMPNN.log" 2>&1
    score_one AbMPNN 1 "AbMPNN zenodo weights via ProteinMPNN code"
  else
    echo "- AbMPNN: weight corrupt or incompatible checkpoint format" >> "$SKIPPED"
  fi
fi

# --- 3 AntiFold ---
echo "[$(date)] AntiFold"
AFW="third_party/AntiFold/models/model.pt"
# AntiFold model is large (~500MB+); wait for stable >= 100MB then try load
if ! wait_file_size "$AFW" 100000000 14400; then
  echo "- AntiFold: weight download incomplete" >> "$SKIPPED"
else
  if "${PYTHON}" -c "import torch; torch.load('${AFW}', map_location='cpu'); print('ok')"; then
    DIR="${OUT_ROOT}/AntiFold"
    mkdir -p "$DIR"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="${PWD}/third_party/AntiFold:${PYTHONPATH}" \
      "${PYTHON}" -u scripts/baselines/run_antifold_nano43.py \
      --samples-json "$SAMPLES" \
      --out-dir "$DIR" \
      --num-seq-per-target 1 \
      --sampling-temp 0.2 \
      > "logs/ext_invfold_AntiFold.log" 2>&1
    score_one AntiFold 1 "AntiFold nanobody+antigen; regions default may need IMGT"
  else
    echo "- AntiFold: weight corrupt/incomplete" >> "$SKIPPED"
  fi
fi

# --- 4 ESM-IF via AntiFold flag ---
echo "[$(date)] ESM-IF"
DIR="${OUT_ROOT}/ESM-IF"
if [[ -d "${OUT_ROOT}/AntiFold" ]] && grep -q "AntiFold: weight" "$SKIPPED"; then
  echo "- ESM-IF: skipped because AntiFold runner/weights unavailable" >> "$SKIPPED"
else
  mkdir -p "$DIR"
  if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="${PWD}/third_party/AntiFold:${PYTHONPATH}" \
      "${PYTHON}" -u scripts/baselines/run_antifold_nano43.py \
      --samples-json "$SAMPLES" \
      --out-dir "$DIR" \
      --esm-if1-mode \
      --num-seq-per-target 1 \
      --sampling-temp 0.2 \
      > "logs/ext_invfold_ESM-IF.log" 2>&1; then
    score_one ESM-IF 1 "ESM-IF1 via AntiFold --esm_if1_mode"
  else
    echo "- ESM-IF: run failed (see logs/ext_invfold_ESM-IF.log)" >> "$SKIPPED"
  fi
fi

# --- 5 LigandMPNN ---
echo "- LigandMPNN: skipped full eval; ProteinMPNN already covers protein Ag as fixed chain (LigandMPNN targets small-molecule/ligand atom context)." >> "$SKIPPED"

# --- 6 AntiBMPNN ---
echo "[$(date)] AntiBMPNN"
if [[ ! -d third_party/AntiBMPNN ]]; then
  if ! git clone --depth 1 https://github.com/zeysun/AntiBMPNN.git third_party/AntiBMPNN \
      > logs/clone_AntiBMPNN.log 2>&1; then
    echo "- AntiBMPNN: clone failed (network)" >> "$SKIPPED"
  fi
fi
if [[ -d third_party/AntiBMPNN ]]; then
  echo "- AntiBMPNN: present but auto-eval adapter not wired yet if weights missing; check Zenodo manually" >> "$SKIPPED"
  # try zenodo weights if README points to them
fi

# --- 7 LM-Design ---
echo "- LM-Design: skipped if ByProt/Zenodo weights not available on this host (download blocked / not configured)." >> "$SKIPPED"

# --- 8 nanoFOLD ---
echo "- nanoFOLD: Google Drive weights blocked on this server (see third_party/nanofold/BLOCKED.md)." >> "$SKIPPED"

"${PYTHON}" scripts/baselines/update_comparison.py --out-root "${OUT_ROOT}"
echo "[$(date)] done"
cat "${OUT_ROOT}/comparison.md"
