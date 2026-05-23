#!/bin/bash
# 5-seed paired AUC validation: degree-adaptive m_v (c=1) vs the
# fixed-m=128 baseline at the Epinions PRODUCTION config.
#
# Acceptance gate (per plan):
#   - Paired-mean AUC within ±0.005 of fixed-m baseline (or HIGHER —
#     the smoke result was +6.7pp at the abbreviated config, so we
#     expect the production config to also lift, not match).
#   - Paired σ within ±1.0σ of baseline (~0.011 from prior memory).
#
# Production config: c2,c3,c4,c5,w2,w3, h=4, 80 epochs, balance
# pruner, edge_cr highway gate, fraction_negative scorer.
#
# Plan: docs/plans/2026-05-10-degree-adaptive-mv/plan.tex
# Smoke report: reports/2026-05-10-degree-adaptive-mv.md
#
# Wall budget: ~2-4 min/seed × 5 seeds × 2 variants ≈ 30-60 min
# total (estimated from prior epinions edge_cr 5-seed run timing).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/adaptive_mv_5seed_2026_05_10"
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/results.tsv"
echo -e "variant\tseed\twall_s\tauc\tf1m" > "$RESULTS"

run_one() {
    local variant="$1" seed="$2"
    local logf="$LOG_DIR/${variant}_seed${seed}.log"
    local t0=$(date +%s)

    if [ "$variant" = "fixed" ]; then
        env_extra=(
            HSIKAN_TOPK_MODE=per_vertex
            HSIKAN_TOPK_K=128
        )
    else
        env_extra=(
            HSIKAN_TOPK_MODE=per_vertex_adaptive
            HSIKAN_TOPK_K=128
            HSIKAN_TOPK_M_V_MIN=1
            HSIKAN_TOPK_M_V_MAX=128
            HSIKAN_TOPK_M_V_C=1.0
        )
    fi

    # Production-config recipe — must match
    # `project_epinions_edge_cr_5seed_2026_05_09` so the fixed-m
    # variant reproduces the published baseline (~0.8464 ± 0.0106).
    # Only the m_v config differs between variants.
    echo "[5seed] $(date +%H:%M:%S) START variant=$variant seed=$seed"
    env "${env_extra[@]}" \
        HYMEKO_CYCLE_CACHE=1 \
        HSIKAN_TOPK_PRUNER=balance \
        HSIKAN_TOPK_SCORER=fraction_negative \
        HSIKAN_TRITON_KERNEL=1 \
        HSIKAN_TRITON_BACKWARD=1 \
        HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_MAX_K2=200000 \
        HSIKAN_MAX_K3=200000 \
        python -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset epinions --hidden 4 --n-epochs 80 \
            --max-k4 100000 --seed "$seed" \
            > "$logf" 2>&1

    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        local auc f1
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        f1=$(echo "$result"  | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["f1m"], 4))')
        echo -e "${variant}\t${seed}\t${elapsed}\t${auc}\t${f1}" >> "$RESULTS"
        echo "[5seed] $(date +%H:%M:%S) OK variant=$variant seed=$seed AUC=$auc wall=${elapsed}s"
    else
        echo "[5seed] $(date +%H:%M:%S) FAIL variant=$variant seed=$seed (see $logf)"
    fi
}

# Run all 10 jobs (interleaving variants per seed so transient
# system load is shared evenly).
for seed in 0 1 2 3 4; do
    run_one fixed    "$seed"
    run_one adaptive "$seed"
done

echo
echo "=== 5-seed paired validation summary ==="
column -t -s $'\t' "$RESULTS"

# Paired stats
python - <<'PY'
import csv, statistics
from collections import defaultdict
with open("/tmp/adaptive_mv_5seed_2026_05_10/results.tsv") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
by = defaultdict(list)
for r in rows:
    by[r["variant"]].append(r)
def m(rs, key, fn=float):
    vs = [fn(r[key]) for r in rs]
    return statistics.mean(vs), statistics.stdev(vs) if len(vs) > 1 else 0.0
print()
print("─── 5-seed paired stats ───")
for variant in ("fixed", "adaptive"):
    rs = sorted(by[variant], key=lambda r: int(r["seed"]))
    if len(rs) != 5:
        print(f"  {variant}: incomplete ({len(rs)}/5)"); continue
    auc_m, auc_s = m(rs, "auc")
    wall_m, wall_s = m(rs, "wall_s")
    aucs = ", ".join(r["auc"] for r in rs)
    print(f"  {variant:<10}  AUC mean={auc_m:.4f} ± {auc_s:.4f}  "
          f"wall mean={wall_m:.0f}s  per-seed: {aucs}")
fixed = sorted(by.get("fixed", []), key=lambda r: int(r["seed"]))
adapt = sorted(by.get("adaptive", []), key=lambda r: int(r["seed"]))
if len(fixed) == 5 and len(adapt) == 5:
    deltas = [float(a["auc"]) - float(f["auc"]) for a, f in zip(adapt, fixed)]
    dm = statistics.mean(deltas)
    ds = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    sigma = dm / ds * (5 ** 0.5) if ds > 0 else float("inf")
    print(f"\n  Paired delta (adaptive - fixed): {dm:+.4f} ± {ds:.4f}  "
          f"σ={sigma:+.2f}")
    print(f"  Per-seed deltas: {', '.join(f'{d:+.4f}' for d in deltas)}")
PY
