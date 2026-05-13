#!/usr/bin/env bash
# Dual-envelope random search: Gömb vs non-HSiKAN AUC bars (SGCN territory).
# Optional Slashdot (compact — node embedding scales with |V|).
# Slashdot joint-mix: RUN_SLASHDOT_JOINT=0 to skip. Default 1 (run_gomb_smoke
# subsamples joint slots on SNAP to fit consumer VRAM).
# Slashdot vanilla: RUN_SLASHDOT_VANILLA=1 (default 1).
# Each smoke JSONL row includes infer_wall_s / infer_edges_per_s (see run_gomb_smoke).
# See: docs/plans/2026-05-12-gomb-external-auc-tuning/plan.tex
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

# Reduce CUDA allocator fragmentation between subprocess trials (PyTorch 2.x).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_BITCOIN="${RUN_BITCOIN:-1}"
RUN_SLASHDOT="${RUN_SLASHDOT:-1}"

# Rank trials by val AUROC so search does not optimise the held-out test set.
PICK_BEST_BY="${PICK_BEST_BY:-val_auroc}"

# More trials ⇒ more hyperparameter draws per envelope (override freely).
TRIALS="${TRIALS:-24}"
NEPOCHS="${NEPOCHS:-72}"
TRIALS_SLASH="${TRIALS_SLASH:-16}"
NEPOCHS_SLASH="${NEPOCHS_SLASH:-48}"
DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-reports}"
TIMEOUT_S="${TIMEOUT_S:-7200}"
TIMEOUT_SLASH="${TIMEOUT_SLASH:-10800}"

mkdir -p "$OUTDIR"

if [[ "$RUN_BITCOIN" == "1" ]]; then
  echo "[gomb-external-auc] joint-mix (Bitcoin) → ${OUTDIR}/gomb_tune_external_joint.jsonl"
  python -m signedkan_wip.src.run_gomb_tune \
    --datasets bitcoin_otc bitcoin_alpha \
    --joint-mix \
    --trials "$TRIALS" \
    --search-seed 11 \
    --data-seed 0 \
    --edge-split 80_10_10 \
    --n-epochs "$NEPOCHS" \
    --device "$DEVICE" \
    --timeout-s "$TIMEOUT_S" \
    --architecture wide \
    --pick-best-by "$PICK_BEST_BY" \
    --out "${OUTDIR}/gomb_tune_external_joint.jsonl"

  echo "[gomb-external-auc] vanilla/mixed (Bitcoin) → ${OUTDIR}/gomb_tune_external_vanilla.jsonl"
  python -m signedkan_wip.src.run_gomb_tune \
    --datasets bitcoin_otc bitcoin_alpha \
    --trials "$TRIALS" \
    --search-seed 13 \
    --data-seed 0 \
    --edge-split 80_10_10 \
    --n-epochs "$NEPOCHS" \
    --device "$DEVICE" \
    --timeout-s "$TIMEOUT_S" \
    --architecture wide \
    --pick-best-by "$PICK_BEST_BY" \
    --out "${OUTDIR}/gomb_tune_external_vanilla.jsonl"
else
  echo "[gomb-external-auc] skipping Bitcoin (RUN_BITCOIN=0)"
fi

if [[ "$RUN_SLASHDOT" == "1" ]]; then
  # Joint-mix on full Slashdot often OOMs on 8GB (four Rust pools + four stacks).
  if [[ "${RUN_SLASHDOT_JOINT:-1}" == "1" ]]; then
    echo "[gomb-external-auc] joint-mix (Slashdot, compact) → ${OUTDIR}/gomb_tune_external_slashdot_joint.jsonl"
    python -m signedkan_wip.src.run_gomb_tune \
      --datasets slashdot \
      --joint-mix \
      --trials "$TRIALS_SLASH" \
      --search-seed 21 \
      --data-seed 0 \
      --edge-split 80_10_10 \
      --n-epochs "$NEPOCHS_SLASH" \
      --device "$DEVICE" \
      --timeout-s "$TIMEOUT_SLASH" \
      --architecture compact \
      --pick-best-by "$PICK_BEST_BY" \
      --out "${OUTDIR}/gomb_tune_external_slashdot_joint.jsonl"
  else
    echo "[gomb-external-auc] skipping Slashdot joint (RUN_SLASHDOT_JOINT=0)"
  fi

  if [[ "${RUN_SLASHDOT_VANILLA:-1}" == "1" ]]; then
    echo "[gomb-external-auc] vanilla/mixed (Slashdot, compact) → ${OUTDIR}/gomb_tune_external_slashdot_vanilla.jsonl"
    python -m signedkan_wip.src.run_gomb_tune \
      --datasets slashdot \
      --trials "$TRIALS_SLASH" \
      --search-seed 23 \
      --data-seed 0 \
      --edge-split 80_10_10 \
      --n-epochs "$NEPOCHS_SLASH" \
      --device "$DEVICE" \
      --timeout-s "$TIMEOUT_SLASH" \
      --architecture compact \
      --pick-best-by "$PICK_BEST_BY" \
      --out "${OUTDIR}/gomb_tune_external_slashdot_vanilla.jsonl"
  else
    echo "[gomb-external-auc] skipping Slashdot vanilla (RUN_SLASHDOT_VANILLA=0)"
  fi
else
  echo "[gomb-external-auc] skipping Slashdot (RUN_SLASHDOT=0)"
fi

echo "[gomb-external-auc] done."
echo "[gomb-external-auc] phase summaries (Markdown):"
python -m signedkan_wip.src.gomb_jsonl_summarize "${OUTDIR}/gomb_tune_external_joint.jsonl" 2>/dev/null || true
python -m signedkan_wip.src.gomb_jsonl_summarize "${OUTDIR}/gomb_tune_external_vanilla.jsonl" 2>/dev/null || true
if [[ "$RUN_SLASHDOT" == "1" ]]; then
  python -m signedkan_wip.src.gomb_jsonl_summarize "${OUTDIR}/gomb_tune_external_slashdot_joint.jsonl" 2>/dev/null || true
  python -m signedkan_wip.src.gomb_jsonl_summarize "${OUTDIR}/gomb_tune_external_slashdot_vanilla.jsonl" 2>/dev/null || true
fi
