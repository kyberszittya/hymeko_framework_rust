#!/bin/bash
# 5-point c-sweep for the degree-adaptive m_v per-vertex enumerator.
#
# Smoke gate (per the plan):
#   AUC ≥ 0.6216 (within ±0.02 of fixed-m baseline 0.6416) AND
#   wall ≤ ~150s end-to-end at any c ∈ {1, 2, 4, 8, 16}.
#
# Plan: docs/plans/2026-05-10-degree-adaptive-mv/plan.tex
#
# Usage: bash signedkan_wip/experiments/run_adaptive_mv_sweep_2026_05_10.sh

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/adaptive_mv_sweep_2026_05_10"
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/results.tsv"
echo -e "c\twall_s\tauc\tf1m\tn_cycles" > "$RESULTS"

for c in 1 2 4 8 16; do
    logf="$LOG_DIR/c_${c}.log"
    echo "[adaptive_mv] $(date +%H:%M:%S) START c=$c"
    t0=$(date +%s)
    HSIKAN_TOPK_MODE=per_vertex_adaptive \
    HSIKAN_TOPK_K=128 \
    HSIKAN_TOPK_M_V_MIN=1 \
    HSIKAN_TOPK_M_V_MAX=128 \
    HSIKAN_TOPK_M_V_C="$c" \
    HSIKAN_TOPK_PRUNER=balance \
    HSIKAN_TOPK_SCORER=fraction_negative \
    HSIKAN_MIXED_TUPLES=c3,c4 \
    python -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 20 --seed 0 \
        > "$logf" 2>&1
    elapsed=$(( $(date +%s) - t0 ))
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        f1=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["f1m"], 4))')
        echo -e "${c}\t${elapsed}\t${auc}\t${f1}\t" >> "$RESULTS"
        echo "[adaptive_mv] $(date +%H:%M:%S) OK c=$c AUC=$auc wall=${elapsed}s"
    else
        echo "[adaptive_mv] $(date +%H:%M:%S) FAIL c=$c (see $logf)"
    fi
done

echo
echo "=== c-sweep summary ==="
column -t -s $'\t' "$RESULTS"
