#!/usr/bin/env bash
# Gömb-strict wikisigned retry — smaller config (topk 64→32, d_embed 32→16)
# to fit on the 8 GB consumer GPU after the v5_combined config OOM'd.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/gomb_wikisigned_retry_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Gömb-strict wikisigned retry (smaller config) ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Wait for the GPU.
echo "[$(date -Is)] Waiting for in-flight GPU jobs..." | tee -a "$LOG"
while pgrep -f "run_gomb_smoke|train_circles_ricci" 2>/dev/null \
        | grep -v $$ | grep -q .; do
    sleep 30
done
echo "[$(date -Is)] GPU clear." | tee -a "$LOG"

# Slimmer config: d_embed 16, topk 32, joint_slot_cap 6000 (vs 12000 default)
# to clear the OOM on the 12K-node wikisigned graph with denser cycles.
ARGS="--d-embed 16 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
      --n-tiers 3 --topk 32 --lr 0.003 --joint-slot-cap 6000"

for seed in 0 1 2 3 4; do
    outf="${OUT_DIR}/wikisigned_seed${seed}.log"
    echo "[$(date -Is)] START seed=$seed" | tee -a "$LOG"
    t0=$(date +%s)
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset wikisigned --seed "$seed" \
        --n-epochs 80 \
        --edge-split 80_10_10 --joint-mix --device cuda \
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

echo | tee -a "$LOG"
echo "=== 5-seed aggregate: wikisigned ===" | tee -a "$LOG"
OUT_DIR_BASH="$OUT_DIR" python3 <<'EOF' | tee -a "$LOG"
import os, json, statistics
from pathlib import Path
out_dir = Path(os.environ["OUT_DIR_BASH"])
aucs = []
for p in sorted(out_dir.glob("wikisigned_seed*.log")):
    with open(p) as f:
        for line in f:
            if line.startswith('{"dataset"'):
                r = json.loads(line)
                aucs.append(r["test_auroc"])
                break
print(f"n_seeds={len(aucs)}")
if aucs:
    print(f"per-seed: {[round(a, 4) for a in aucs]}")
    print(f"mean = {statistics.mean(aucs):.4f}")
    if len(aucs) > 1:
        print(f"pstd = {statistics.pstdev(aucs):.4f}")
EOF

echo "end: $(date -Is)" | tee -a "$LOG"
echo "results in: $OUT_DIR" | tee -a "$LOG"
