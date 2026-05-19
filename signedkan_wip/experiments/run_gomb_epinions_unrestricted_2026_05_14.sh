#!/usr/bin/env bash
# Gömb-UNRESTRICTED (transductive) Epinions — 5-seed run.
#
# Purpose: bracket the strict 0.9526 ± 0.0018 number from above by
# running the same v5_combined config under the canonical transductive
# convention (test-edge signs participate in cycle σ-products).
#
# v5_combined config (the strict-SOTA winner):
#   --d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8
#   --n-tiers 3 --topk 64 --lr 0.003 --n-epochs 80 --joint-mix
#   --edge-split 80_10_10 --device cuda
#
# Plan: docs/plans/2026-05-14-gomb-unrestricted/

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/gomb_epinions_unrestricted_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Gömb-UNRESTRICTED Epinions 5-seed ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "host: $(uname -a)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

ARGS="--d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
      --n-tiers 3 --topk 64 --lr 0.003"

for seed in 0 1 2 3 4; do
    outf="${OUT_DIR}/unrestricted_v5_combined_seed${seed}.log"
    echo "[$(date -Is)] START seed=$seed" | tee -a "$LOG"
    t0=$(date +%s)
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset epinions --seed "$seed" \
        --n-epochs 80 \
        --edge-split 80_10_10 --joint-mix --device cuda \
        --unrestricted-cycles \
        $ARGS \
        > "$outf" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    result=$(grep -E '^\{"dataset"' "$outf" | tail -1)
    auc=$(echo "$result" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    print(f'val={r.get(\"val_auroc\", \"?\"):.4f} test={r.get(\"test_auroc\", \"?\"):.4f}')
except Exception:
    print('(no result)')
")
    echo "[$(date -Is)] DONE  seed=$seed rc=$rc elapsed=${elapsed}s $auc" | tee -a "$LOG"
done

# Aggregate.
echo | tee -a "$LOG"
echo "=== 5-seed aggregate ===" | tee -a "$LOG"
python3 <<EOF | tee -a "$LOG"
import json, statistics
from pathlib import Path
out_dir = Path("$OUT_DIR")
aucs = []
for p in sorted(out_dir.glob("unrestricted_v5_combined_seed*.log")):
    with open(p) as f:
        for line in f:
            if line.startswith('{"dataset"'):
                r = json.loads(line)
                aucs.append(r["test_auroc"])
                break
print(f"n_seeds={len(aucs)}")
print(f"test_auroc per seed: {[round(a, 4) for a in aucs]}")
if aucs:
    print(f"mean = {statistics.mean(aucs):.4f}")
    if len(aucs) > 1:
        print(f"pstd = {statistics.pstdev(aucs):.4f}")
EOF

echo "end: $(date -Is)" | tee -a "$LOG"
echo "results in: $OUT_DIR" | tee -a "$LOG"
