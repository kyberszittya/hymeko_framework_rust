#!/usr/bin/env bash
# Gömb-strict on wiki_elec + wikisigned + wiki_conflict — 5 seeds each.
#
# Purpose: extend the canonical 4-dataset signed-link benchmark suite
# (Bitcoin Alpha / OTC / Slashdot / Epinions) to 6-7 datasets by adding
# the Wikipedia signed networks. wiki_elec and wikisigned were
# referenced in the Konect literature alongside the SNAP datasets;
# they are the 5th and 6th canonical benchmark.
#
# Config: same v5_combined config as the Epinions strict-SOTA run.
# Strict protocol throughout (cycle pool on train edges only). 80 epochs.
#
# Plan: docs/plans/2026-05-14-gomb-unrestricted/ (extends the same
# Gömb deployment line — additional datasets, no architectural change).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/gomb_wiki_datasets_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Gömb-strict on wiki_elec + wikisigned ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Wait for the GPU to clear.
echo "[$(date -Is)] Waiting for in-flight Gömb / HymeYOLO to clear..." \
    | tee -a "$LOG"
while pgrep -f "run_gomb_smoke|run_gomb_epinions_unrestricted|train_circles_ricci" \
        2>/dev/null | grep -v $$ | grep -q .; do
    sleep 30
done
echo "[$(date -Is)] GPU clear. Beginning wiki-datasets sweep." | tee -a "$LOG"

# v5_combined: same config that produced Epinions strict 0.9526.
ARGS="--d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
      --n-tiers 3 --topk 64 --lr 0.003"

run_one() {
    local dataset="$1"; local seed="$2"; local epochs="$3"
    local outf="${OUT_DIR}/${dataset}_seed${seed}.log"
    echo "[$(date -Is)] START $dataset seed=$seed epochs=$epochs" | tee -a "$LOG"
    local t0=$(date +%s)
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset "$dataset" --seed "$seed" \
        --n-epochs "$epochs" \
        --edge-split 80_10_10 --joint-mix --device cuda \
        $ARGS \
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
    echo "[$(date -Is)] DONE  $dataset seed=$seed rc=$rc elapsed=${elapsed}s $auc" \
        | tee -a "$LOG"
}

# 5 seeds × 2 datasets (wiki_elec ~7k nodes, wikisigned ~12k nodes).
# Bigger wiki_conflict (~118k nodes) deferred: would need own slot-cap
# tuning, and the present goal is a clean 5th + 6th dataset.
for dataset in wiki_elec wikisigned; do
    for seed in 0 1 2 3 4; do
        run_one "$dataset" "$seed" 80
    done
    # Aggregate this dataset.
    echo | tee -a "$LOG"
    echo "=== 5-seed aggregate: $dataset ===" | tee -a "$LOG"
    DATASET="$dataset" OUT_DIR_BASH="$OUT_DIR" python3 <<'EOF' | tee -a "$LOG"
import os, json, statistics
from pathlib import Path
dataset = os.environ["DATASET"]
out_dir = Path(os.environ["OUT_DIR_BASH"])
aucs = []
for p in sorted(out_dir.glob(f"{dataset}_seed*.log")):
    with open(p) as f:
        for line in f:
            if line.startswith('{"dataset"'):
                r = json.loads(line)
                aucs.append(r["test_auroc"])
                break
print(f"dataset={dataset} n_seeds={len(aucs)}")
if aucs:
    print(f"  per-seed: {[round(a, 4) for a in aucs]}")
    print(f"  mean = {statistics.mean(aucs):.4f}")
    if len(aucs) > 1:
        print(f"  pstd = {statistics.pstdev(aucs):.4f}")
EOF
    echo | tee -a "$LOG"
done

echo "end: $(date -Is)" | tee -a "$LOG"
echo "results in: $OUT_DIR" | tee -a "$LOG"
