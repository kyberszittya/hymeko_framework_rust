#!/usr/bin/env bash
# Gömb-strict Bitcoin architectural levers — 5 single-knob variations
# of the strict-protocol baseline. No Optuna; manual lever picks.
#
# Per-dataset baselines (from run_gomb_strict_benchmark_2026_05_14.sh):
#   Alpha: M_outer=8 d_outer=20 d_middle=24 d_core=48 n_tiers=4 topk=56 lr=5e-3
#   OTC:   M_outer=12 d_outer=8 d_middle=16 d_core=32 n_tiers=2 topk=32 lr=5e-3
#
# Levers (each applied independently to the per-dataset baseline):
#   L1 c5_arity    — add c5 to the cycle mix
#   L2 deeper      — n_tiers ×2 (Alpha 4→8, OTC 2→4)
#   L3 wider_outer — M_outer ×2 (Alpha 8→16, OTC 12→24)
#   L4 long_train  — n_epochs 80→200
#   L5 big_embed   — d_embed 32→64
#
# Plan: docs/plans/2026-05-14-gomb-bitcoin-strict-levers/

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/gomb_bitcoin_levers_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Gömb-strict Bitcoin architectural levers ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Wait for any in-flight GPU job.
echo "[$(date -Is)] Waiting for in-flight GPU jobs to clear..." | tee -a "$LOG"
while pgrep -f "run_gomb_smoke|train_circles_ricci" 2>/dev/null \
        | grep -v $$ | grep -q .; do
    sleep 30
done
echo "[$(date -Is)] GPU clear." | tee -a "$LOG"

# Per-dataset baseline config (no --joint-mix here; baselines are
# single-arity joint configs that produced the strict-benchmark numbers).
ALPHA_BASE="--M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 \
            --n-tiers 4 --topk 56 --lr 0.005 --joint-mix"
OTC_BASE="--M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 \
          --n-tiers 2 --topk 32 --lr 0.005 --joint-mix"

run_lever() {
    local label="$1"; shift
    local dataset="$1"; shift
    local seed="$1"; shift
    local epochs="$1"; shift
    local extra_args="$@"
    local outf="${OUT_DIR}/${dataset}_${label}_seed${seed}.log"
    echo "[$(date -Is)] START ${dataset} ${label} seed=$seed epochs=$epochs" \
        | tee -a "$LOG"
    local t0=$(date +%s)
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset "$dataset" --seed "$seed" \
        --n-epochs "$epochs" \
        --edge-split 80_10_10 --device cuda \
        $extra_args \
        > "$outf" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$outf" | tail -1)
    local auc
    auc=$(echo "$result" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    print(f'val={r.get(\"val_auroc\", \"?\"):.4f} test={r.get(\"test_auroc\", \"?\"):.4f}')
except Exception:
    print('(no result)')
")
    echo "[$(date -Is)] DONE  ${dataset} ${label} seed=$seed rc=$rc elapsed=${elapsed}s $auc" \
        | tee -a "$LOG"
}

# =============================================================
# Bitcoin Alpha — 5 levers × 5 seeds each
# =============================================================
for seed in 0 1 2 3 4; do
    # L0 baseline (sanity-replication)
    run_lever "L0_baseline" "bitcoin_alpha" "$seed" 80 $ALPHA_BASE

    # L1 c5_arity
    run_lever "L1_c5" "bitcoin_alpha" "$seed" 80 $ALPHA_BASE --cycle-ks 3,4,5

    # L2 deeper
    run_lever "L2_deeper" "bitcoin_alpha" "$seed" 80 \
        --M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 \
        --n-tiers 8 --topk 56 --lr 0.005 --joint-mix

    # L3 wider_outer
    run_lever "L3_wider" "bitcoin_alpha" "$seed" 80 \
        --M-outer 16 --d-outer 20 --d-middle 24 --d-core 48 \
        --n-tiers 4 --topk 56 --lr 0.005 --joint-mix

    # L4 long_train
    run_lever "L4_long" "bitcoin_alpha" "$seed" 200 $ALPHA_BASE

    # L5 big_embed
    run_lever "L5_big_embed" "bitcoin_alpha" "$seed" 80 \
        --d-embed 64 \
        --M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 \
        --n-tiers 4 --topk 56 --lr 0.005 --joint-mix
done

# =============================================================
# Bitcoin OTC — 5 levers × 5 seeds each
# =============================================================
for seed in 0 1 2 3 4; do
    run_lever "L0_baseline" "bitcoin_otc" "$seed" 80 $OTC_BASE
    run_lever "L1_c5" "bitcoin_otc" "$seed" 80 $OTC_BASE --cycle-ks 3,4,5
    run_lever "L2_deeper" "bitcoin_otc" "$seed" 80 \
        --M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 \
        --n-tiers 4 --topk 32 --lr 0.005 --joint-mix
    run_lever "L3_wider" "bitcoin_otc" "$seed" 80 \
        --M-outer 24 --d-outer 8 --d-middle 16 --d-core 32 \
        --n-tiers 2 --topk 32 --lr 0.005 --joint-mix
    run_lever "L4_long" "bitcoin_otc" "$seed" 200 $OTC_BASE
    run_lever "L5_big_embed" "bitcoin_otc" "$seed" 80 \
        --d-embed 64 \
        --M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 \
        --n-tiers 2 --topk 32 --lr 0.005 --joint-mix
done

# =============================================================
# Aggregate
# =============================================================
echo | tee -a "$LOG"
echo "=== Aggregate (5-seed mean / pstd per dataset × lever) ===" | tee -a "$LOG"
OUT_DIR_BASH="$OUT_DIR" python3 <<'EOF' | tee -a "$LOG"
import os, json, statistics
from collections import defaultdict
from pathlib import Path
out_dir = Path(os.environ["OUT_DIR_BASH"])
by_key = defaultdict(list)
for p in sorted(out_dir.glob("*.log")):
    if "orchestrator" in p.name:
        continue
    parts = p.stem.split("_")
    if len(parts) < 4:
        continue
    # e.g., bitcoin_alpha_L0_baseline_seed0
    dataset = "_".join(parts[:2])
    lever = "_".join(parts[2:-1])
    with open(p) as f:
        for line in f:
            if line.startswith('{"dataset"'):
                r = json.loads(line)
                by_key[(dataset, lever)].append(r["test_auroc"])
                break

for (ds, lv), aucs in sorted(by_key.items()):
    if aucs:
        m = statistics.mean(aucs)
        s = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
        print(f"  {ds:13} {lv:14} n={len(aucs)} mean={m:.4f} pstd={s:.4f}")
    else:
        print(f"  {ds:13} {lv:14} n=0 (no results)")
EOF

echo "end: $(date -Is)" | tee -a "$LOG"
echo "results in: $OUT_DIR" | tee -a "$LOG"
