#!/usr/bin/env bash
#
# m_v adaptive speedup smoke (2026-06-04) — local Bitcoin Alpha probe
# before deploying narrower m_v on Komondor Epinions cold-cache.
#
# Question: how much AUC do we lose by tightening per-vertex m_v?
# Baseline: HSIKAN_TOPK_K=128 fixed (the current production cap).
# Variants: degree-adaptive m_v with c ∈ {1.0, 0.5, 0.25}
#   (m_v_min=1, m_v_max=128).
#
# Decision rule (per memory `feedback_n_seed_before_paper_promotion`
# and `feedback_no_structure_ablation_first`):
#   - AUC within ±0.005 of fixed-128 at n=1 -> escalate to 3-seed
#   - 3-seed mean within ±0.005 -> safe to deploy on Epinions
#   - Otherwise: report the AUC vs wall trade-off, defer decision
#
# Wall: ~30s/cell × 4 configs = ~2 min total on Bitcoin Alpha.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/mv_speedup_smoke_2026_06_04"
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/results.tsv"
echo -e "variant\tseed\twall_s\tauc\tf1m\tn_params" > "$RESULTS"

run_one() {
    local variant="$1" seed="$2"
    local logf="$LOG_DIR/${variant}_seed${seed}.log"
    local -a env_extra=()

    case "$variant" in
        fixed_128)
            env_extra=(
                HSIKAN_TOPK_MODE=per_vertex
                HSIKAN_TOPK_K=128
            )
            ;;
        adaptive_c1)
            env_extra=(
                HSIKAN_TOPK_MODE=per_vertex_adaptive
                HSIKAN_TOPK_K=128
                HSIKAN_TOPK_M_V_MIN=1
                HSIKAN_TOPK_M_V_MAX=128
                HSIKAN_TOPK_M_V_C=1.0
            )
            ;;
        adaptive_c05)
            env_extra=(
                HSIKAN_TOPK_MODE=per_vertex_adaptive
                HSIKAN_TOPK_K=128
                HSIKAN_TOPK_M_V_MIN=1
                HSIKAN_TOPK_M_V_MAX=128
                HSIKAN_TOPK_M_V_C=0.5
            )
            ;;
        adaptive_c025)
            env_extra=(
                HSIKAN_TOPK_MODE=per_vertex_adaptive
                HSIKAN_TOPK_K=128
                HSIKAN_TOPK_M_V_MIN=1
                HSIKAN_TOPK_M_V_MAX=128
                HSIKAN_TOPK_M_V_C=0.25
            )
            ;;
        *) echo "unknown variant $variant" >&2; return 1 ;;
    esac

    local t0=$(date +%s)
    echo "[mv-smoke] $(date +%H:%M:%S) START variant=$variant seed=$seed"

    env "${env_extra[@]}" \
        HYMEKO_CYCLE_CACHE=1 \
        HYMEKO_CYCLE_CACHE_DIR="$LOG_DIR/cache_$variant" \
        HSIKAN_TOPK_PRUNER=balance \
        HSIKAN_TOPK_SCORER=fraction_negative \
        HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_MAX_K2=200000 \
        HSIKAN_MAX_K3=200000 \
        PYTHONPATH=. /home/kyberszittya/miniconda3/bin/python \
            -m hymeko_neuro.experiments.runs.run_final_cell \
            --dataset bitcoin_alpha --hidden 4 --n-epochs 80 \
            --max-k4 200000 --seed "$seed" \
            > "$logf" 2>&1

    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        local auc f1 np
        auc=$(echo "$result" | /home/kyberszittya/miniconda3/bin/python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        f1=$(echo "$result"  | /home/kyberszittya/miniconda3/bin/python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["f1m"], 4))')
        np=$(echo "$result"  | /home/kyberszittya/miniconda3/bin/python -c 'import sys,json;print(json.loads(sys.stdin.read())["n_params"])')
        echo -e "${variant}\t${seed}\t${elapsed}\t${auc}\t${f1}\t${np}" >> "$RESULTS"
        echo "[mv-smoke] $(date +%H:%M:%S) OK variant=$variant seed=$seed AUC=$auc wall=${elapsed}s"
    else
        echo "[mv-smoke] $(date +%H:%M:%S) FAIL variant=$variant seed=$seed rc=$rc (see $logf)"
        tail -3 "$logf" | sed 's/^/    /'
    fi
}

SEEDS="${SEEDS:-0}"   # default single seed; override with SEEDS="0 1 2"
for seed in $SEEDS; do
    for v in fixed_128 adaptive_c1 adaptive_c05 adaptive_c025; do
        run_one "$v" "$seed"
    done
done

echo
echo "=== m_v speedup smoke summary ==="
column -t -s $'\t' "$RESULTS"
echo
echo "Wall + AUC delta vs fixed_128 baseline:"
/home/kyberszittya/miniconda3/bin/python - <<PY
import csv, statistics
from collections import defaultdict
rows = list(csv.DictReader(open("$RESULTS"), delimiter="\t"))
by_var = defaultdict(list)
for r in rows:
    by_var[r["variant"]].append((float(r["wall_s"]), float(r["auc"])))
if "fixed_128" not in by_var:
    print("no baseline; aborting")
else:
    bw = statistics.mean(w for w,_ in by_var["fixed_128"])
    ba = statistics.mean(a for _,a in by_var["fixed_128"])
    print(f"  fixed_128   : wall={bw:.1f}s auc={ba:.4f} (baseline)")
    for v, vs in by_var.items():
        if v == "fixed_128": continue
        w = statistics.mean(x for x,_ in vs)
        a = statistics.mean(x for _,x in vs)
        ws = (w-bw)/bw*100
        as_ = a - ba
        print(f"  {v:<12}: wall={w:.1f}s ({ws:+.1f}%) auc={a:.4f} ({as_:+.4f})")
PY
