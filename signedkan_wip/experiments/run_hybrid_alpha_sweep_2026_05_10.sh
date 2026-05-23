#!/bin/bash
# 5-point α-sweep for the hybrid scorer at the abbreviated Epinions
# smoke-test config (c3+c4, h=4, 20 epochs, K=10000, balance pruner,
# signal=fraction_negative, heuristic=entropy).
#
# Pass criterion: any α achieves AUC ≥ 0.6216 (per-vertex baseline
# 0.6416 ± 0.02).  See
# docs/plans/2026-05-10-hybrid-alpha-scorer/plan.tex (smoke gate).
#
# Usage: bash signedkan_wip/experiments/run_hybrid_alpha_sweep_2026_05_10.sh

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/hybrid_alpha_sweep_2026_05_10"
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/results.tsv"
echo -e "alpha\twall_s\tauc\tf1m\tn_cycles" > "$RESULTS"

for alpha in 0.0 0.25 0.5 0.75 1.0; do
    logf="$LOG_DIR/alpha_${alpha}.log"
    echo "[hybrid] $(date +%H:%M:%S) START alpha=$alpha"
    t0=$(date +%s)
    HSIKAN_TOPK_MODE=entropy \
    HSIKAN_TOPK_K=10000 \
    HSIKAN_TOPK_PRUNER=balance \
    HSIKAN_TOPK_HEURISTIC=entropy \
    HSIKAN_TOPK_HYBRID_ALPHA="$alpha" \
    HSIKAN_TOPK_SIGNAL=fraction_negative \
    HSIKAN_MIXED_TUPLES=c3,c4 \
    python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 20 --seed 0 \
        > "$logf" 2>&1
    elapsed=$(( $(date +%s) - t0 ))
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        f1=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["f1m"], 4))')
        echo -e "${alpha}\t${elapsed}\t${auc}\t${f1}\t10000" >> "$RESULTS"
        echo "[hybrid] $(date +%H:%M:%S) OK alpha=$alpha AUC=$auc wall=${elapsed}s"
    else
        echo "[hybrid] $(date +%H:%M:%S) FAIL alpha=$alpha (see $logf)"
    fi
done

echo
echo "=== α-sweep summary ==="
column -t -s $'\t' "$RESULTS"
