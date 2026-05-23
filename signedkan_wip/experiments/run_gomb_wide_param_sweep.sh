#!/usr/bin/env bash
# Wide Gömb hyperparameter coverage: many trials × several independent
# ``--search-seed`` streams (different RNG draws from the same discrete menus
# in ``run_gomb_tune.sample_params`` / ``for_joint_mix_tuning``).
#
# Usage (from repo root, or any cwd — script cds to root):
#   bash signedkan_wip/experiments/run_gomb_wide_param_sweep.sh
#
# Environment (all optional):
#   DATASETS       — default: bitcoin_otc bitcoin_alpha
#   TRIALS         — trials per (dataset block) per search seed; default 32
#   NEPOCHS        — default 72
#   SEARCH_SEEDS   — space-separated; default: 11 17 23 31 41 53
#   DEVICE         — default cuda
#   OUT            — single JSONL append target; default reports/gomb_wide_sweep.jsonl
#   JOINT          — 1 = joint-mix passes (default 1)
#   VANILLA        — 1 = vanilla/mixed passes (default 0; set 1 for second loop)
#   PICK_BEST_BY   — default val_auroc
#   ARCHITECTURE   — default wide (use compact for SNAP-heavy graphs)
#
# Example (Slashdot compact joint, many trials, three RNG streams):
#   DATASETS=slashdot ARCHITECTURE=compact JOINT=1 VANILLA=0 TRIALS=20 \\
#     SEARCH_SEEDS="7 13 19" NEPOCHS=48 OUT=reports/slash_sweep.jsonl \\
#     bash signedkan_wip/experiments/run_gomb_wide_param_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PICK_BEST_BY="${PICK_BEST_BY:-val_auroc}"
DATASETS="${DATASETS:-bitcoin_otc bitcoin_alpha}"
read -r -a DS <<< "${DATASETS}"
TRIALS="${TRIALS:-32}"
NEPOCHS="${NEPOCHS:-72}"
DEVICE="${DEVICE:-cuda}"
TIMEOUT_S="${TIMEOUT_S:-7200}"
OUT="${OUT:-reports/gomb_wide_sweep.jsonl}"
ARCHITECTURE="${ARCHITECTURE:-wide}"
JOINT="${JOINT:-1}"
VANILLA="${VANILLA:-0}"
# shellcheck disable=SC2206
SEARCH_SEEDS=(${SEARCH_SEEDS:-11 17 23 31 41 53})

mkdir -p "$(dirname "$OUT")"

_run_tune() {
  local joint_flag=()
  if [[ "$1" == "joint" ]]; then
    joint_flag=(--joint-mix)
  fi
  local ss="$2"
  echo "[gomb-wide-sweep] ${1} search-seed=${ss} trials=${TRIALS} → ${OUT}"
  python -m signedkan_wip.experiments.runs.run_gomb_tune \
    --datasets "${DS[@]}" \
    "${joint_flag[@]}" \
    --trials "$TRIALS" \
    --search-seed "$ss" \
    --data-seed "${DATA_SEED:-0}" \
    --edge-split "${EDGE_SPLIT:-80_10_10}" \
    --n-epochs "$NEPOCHS" \
    --device "$DEVICE" \
    --timeout-s "$TIMEOUT_S" \
    --architecture "$ARCHITECTURE" \
    --pick-best-by "$PICK_BEST_BY" \
    --out "$OUT"
}

for ss in "${SEARCH_SEEDS[@]}"; do
  if [[ "$JOINT" == "1" ]]; then
    _run_tune joint "$ss"
  fi
done

for ss in "${SEARCH_SEEDS[@]}"; do
  if [[ "$VANILLA" == "1" ]]; then
    _run_tune vanilla "$ss"
  fi
done

echo "[gomb-wide-sweep] done. Phase summaries:"
python -m signedkan_wip.src.gomb_jsonl_summarize "$OUT" 2>/dev/null || true
