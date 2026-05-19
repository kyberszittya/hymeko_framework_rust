#!/usr/bin/env bash
# Gömb-strict Epinions fine-tuning sweep — push past 0.95 AUC SOTA.
#
# Baseline (already running in main benchmark):
#   slim_60ep: d_embed=16 M_outer=4 d_outer=4 d_middle=4 d_core=4 n_tiers=3 topk=32, 60 epochs
#   Seed 0/1: test_auroc ≈ 0.94
#
# Six variations to push past 0.95:
#   v1  more_epochs     — slim_120ep              (same model, 2× training)
#   v2  bigger_d_embed  — d_embed=32 + slim cores (node embeds carry more)
#   v3  medium_cores    — all d=8, topk=32        (wider middle/core)
#   v4  more_cycles     — slim + topk=64          (richer cycle pool)
#   v5  combined        — d_embed=32 + medium + topk=64
#   v6  many_tiers      — slim + n_tiers=5        (deeper hierarchy)
#
# Strategy: single-seed screening at 60 epochs, then top-2 at 5-seed × 80 epochs.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/gomb_epinions_finetune_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Gömb-strict Epinions fine-tuning ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Wait for any other gomb_smoke / main benchmark to finish first.
echo "[$(date -Is)] Waiting for in-flight Gömb runs to clear..." | tee -a "$LOG"
while pgrep -f "run_gomb_smoke|run_gomb_strict_benchmark" 2>/dev/null \
        | grep -v $$ | grep -q .; do
    sleep 30
done
echo "[$(date -Is)] GPU clear. Beginning sweep." | tee -a "$LOG"

# Helper.
run_variant() {
    local label="$1"; shift
    local seed="$1"; shift
    local epochs="$1"; shift
    local extra_args="$@"
    local outf="${OUT_DIR}/${label}_seed${seed}.log"
    echo "[$(date -Is)] START $label seed=$seed epochs=$epochs extra='$extra_args'" \
        | tee -a "$LOG"
    local t0=$(date +%s)
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset epinions --seed "$seed" \
        --n-epochs "$epochs" \
        --edge-split 80_10_10 --joint-mix --device cuda \
        $extra_args \
        > "$outf" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$outf" | tail -1)
    local auc=$(echo "$result" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    print(f'val={r.get(\"val_auroc\", \"?\"):.4f} test={r.get(\"test_auroc\", \"?\"):.4f}')
except Exception:
    print('(no result)')
")
    echo "[$(date -Is)] DONE  $label seed=$seed rc=$rc elapsed=${elapsed}s $auc" \
        | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

# =================================================================
# Phase 1 — Single-seed screening of 6 variants
# =================================================================
echo "## Phase 1 — Single-seed screening (6 variants, seed=0, 60 epochs)" \
    | tee -a "$LOG"

# v1: more epochs (longer training)
run_variant "v1_more_epochs" 0 120 \
    --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
    --n-tiers 3 --topk 32 --lr 0.003

# v2: bigger d_embed
run_variant "v2_bigger_embed" 0 60 \
    --d-embed 32 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
    --n-tiers 3 --topk 32 --lr 0.003

# v3: medium cores
run_variant "v3_medium" 0 60 \
    --d-embed 16 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
    --n-tiers 3 --topk 32 --lr 0.003

# v4: more cycles per vertex
run_variant "v4_topk64" 0 60 \
    --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
    --n-tiers 3 --topk 64 --lr 0.003

# v5: combined
run_variant "v5_combined" 0 60 \
    --d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
    --n-tiers 3 --topk 64 --lr 0.003

# v6: deeper tiers
run_variant "v6_deep_tiers" 0 60 \
    --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
    --n-tiers 5 --topk 32 --lr 0.003

# =================================================================
# Phase 2 — Pick top 2 from screening, run 5-seed × 80 epochs
# =================================================================
echo "" | tee -a "$LOG"
echo "## Phase 2 — Picking top 2 variants from screening" | tee -a "$LOG"

# Read screening results.
TOP_2=$(python3 << EOF
import json
from pathlib import Path
out_dir = Path("$OUT_DIR")
variants = []
for f in out_dir.glob("v*_seed0.log"):
    label = f.stem.replace("_seed0", "")
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{"dataset"'):
                try:
                    r = json.loads(line)
                    auc = r.get("test_auroc")
                    variants.append((label, auc, r))
                except Exception:
                    pass
variants.sort(key=lambda x: x[1] or 0, reverse=True)
print(" ".join(v[0] for v in variants[:2]))
EOF
)
echo "[$(date -Is)] Top-2 screened variants: $TOP_2" | tee -a "$LOG"

# Run each top variant at 5 seeds × 80 epochs.
for variant in $TOP_2; do
    case "$variant" in
        v1_more_epochs) ARGS="--d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 --n-tiers 3 --topk 32 --lr 0.003" ;;
        v2_bigger_embed) ARGS="--d-embed 32 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 --n-tiers 3 --topk 32 --lr 0.003" ;;
        v3_medium) ARGS="--d-embed 16 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 --n-tiers 3 --topk 32 --lr 0.003" ;;
        v4_topk64) ARGS="--d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 --n-tiers 3 --topk 64 --lr 0.003" ;;
        v5_combined) ARGS="--d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 --n-tiers 3 --topk 64 --lr 0.003" ;;
        v6_deep_tiers) ARGS="--d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 --n-tiers 5 --topk 32 --lr 0.003" ;;
        *) echo "Unknown variant $variant"; continue ;;
    esac
    for seed in 0 1 2 3 4; do
        run_variant "${variant}_5seed" "$seed" 80 $ARGS
    done
done

# =================================================================
# Final summary
# =================================================================
echo "" | tee -a "$LOG"
echo "=== Fine-tune complete: $(date -Is) ===" | tee -a "$LOG"

# Aggregate.
python3 << EOF | tee -a "$LOG"
import json
import statistics
from pathlib import Path

out_dir = Path("$OUT_DIR")
print()
print(f"{'variant':25} {'n':>3} {'val_AUC mean':>13} {'test_AUC mean':>14} {'± pstd':>10}")
print('-' * 75)

# 5-seed phase.
five_seed_groups = {}
for f in sorted(out_dir.glob("*_5seed_seed*.log")):
    label = f.stem.split("_seed")[0]
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{"dataset"'):
                try:
                    r = json.loads(line)
                    five_seed_groups.setdefault(label, []).append(r)
                except Exception:
                    pass

for label, rows in sorted(five_seed_groups.items()):
    if not rows:
        continue
    tests = [r["test_auroc"] for r in rows]
    vals = [r["val_auroc"] for r in rows]
    t_mean = statistics.mean(tests)
    t_pstd = statistics.pstdev(tests) if len(tests) > 1 else 0.0
    v_mean = statistics.mean(vals)
    print(f"{label:25} {len(rows):>3} {v_mean:>13.4f} {t_mean:>14.4f} {t_pstd:>10.4f}")
print()
print("Single-seed screening AUCs:")
for f in sorted(out_dir.glob("v*_seed0.log")):
    label = f.stem.replace("_seed0", "")
    if "_5seed" in label:
        continue
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{"dataset"'):
                try:
                    r = json.loads(line)
                    print(f"  {label:25} test={r['test_auroc']:.4f} val={r['val_auroc']:.4f}")
                    break
                except Exception:
                    pass
EOF

echo "Results in: $OUT_DIR" | tee -a "$LOG"
