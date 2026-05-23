#!/usr/bin/env bash
# Overnight Gömb hyperparameter sweep (random search, JSONL append).
# Order: **Bitcoin first** → Slashdot → Epinions (compact / low-width) → SBM.
#
# SOTA reference (test ROC-AUC, ``signedkan_wip/src/benchmarks/sota_reference.json``):
#   Slashdot: HSiKAN headline ~0.861, SGT ~0.897
#   Epinions: HSiKAN lower-bound cell ~0.606, SGT ~0.941
# ``compact`` trials bias to small ``d_embed`` / few FIR banks so total learnable
# params sit **well below** wide Gömb (~order 1/8 ballpark vs Bitcoin-wide smoke);
# Slashdot/Epinions are still dominated by ``|V|·d_embed`` — this is an aspirational
# search, not a proof of strict param-ratio vs HSiKAN FLOPs.
#
# Run from repo root:
#   nohup bash signedkan_wip/experiments/run_gomb_tune_overnight.sh \
#     >> reports/gomb_tune_overnight.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="reports/gomb_tune_${STAMP}.jsonl"
LOG="reports/gomb_tune_${STAMP}.log"
echo "[run_gomb_tune_overnight] OUT=${OUT}" | tee -a "${LOG}"

PY="${PYTHON:-python3}"

# Phase 1 — Bitcoin only (wide search; finish before large SNAP graphs).
"${PY}" -m signedkan_wip.experiments.runs.run_gomb_tune \
  --datasets bitcoin_alpha bitcoin_otc \
  --trials 28 --search-seed 101 --data-seed 0 \
  --architecture wide \
  --edge-split 80_10_10 --n-epochs 110 --device cuda --timeout-s 9000 \
  --out "${OUT}" 2>&1 | tee -a "${LOG}"

# Phase 2 — Slashdot (compact widths; more trials; long timeout per trial).
"${PY}" -m signedkan_wip.experiments.runs.run_gomb_tune \
  --datasets slashdot \
  --trials 22 --search-seed 202 --data-seed 0 \
  --architecture compact \
  --edge-split 80_10_10 --n-epochs 76 --device cuda --timeout-s 18000 \
  --out "${OUT}" 2>&1 | tee -a "${LOG}"

# Phase 3 — Epinions (compact; fewer epochs; heaviest enumerator).
"${PY}" -m signedkan_wip.experiments.runs.run_gomb_tune \
  --datasets epinions \
  --trials 14 --search-seed 303 --data-seed 0 \
  --architecture compact \
  --edge-split 80_10_10 --n-epochs 44 --device cuda --timeout-s 28800 \
  --out "${OUT}" 2>&1 | tee -a "${LOG}"

# Phase 4 — SBM (wide; quick sanity vs sota_reference sbm_* rows).
"${PY}" -m signedkan_wip.experiments.runs.run_gomb_tune \
  --datasets sbm_n200 sbm_n400 \
  --trials 24 --search-seed 404 --data-seed 0 \
  --architecture wide \
  --edge-split 80_10_10 --n-epochs 110 --device cuda --timeout-s 9000 \
  --out "${OUT}" 2>&1 | tee -a "${LOG}"

echo "[run_gomb_tune_overnight] finished OUT=${OUT}" | tee -a "${LOG}"
