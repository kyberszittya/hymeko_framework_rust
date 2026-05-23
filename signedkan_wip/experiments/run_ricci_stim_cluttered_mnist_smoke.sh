#!/usr/bin/env bash
# Ricci-Stim Cluttered MNIST smoke — small training run to verify the
# pipeline runs end-to-end on real data before queuing the full
# falsification battery.
#
# 50 train images × 2 epochs × 1 seed × 1 config (E: full Bochner + SDRF)
# Expected wall: < 5 min on the RTX 2070 SUPER.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/ricci_stim_smoke_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/smoke.log"

echo "=== Ricci-Stim Cluttered MNIST smoke ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

python -m signedkan_wip.experiments.run_ricci_stim_cluttered_mnist \
    --n-train 50 --n-eval 20 --n-epochs 2 --seed 0 \
    --config E --device cuda \
    --out-jsonl "$OUT_DIR/smoke.jsonl" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "end: $(date -Is)" | tee -a "$LOG"
echo "results in: $OUT_DIR" | tee -a "$LOG"
