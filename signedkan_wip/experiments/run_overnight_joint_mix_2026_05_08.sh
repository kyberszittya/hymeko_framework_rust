#!/bin/bash
# Overnight 5-seed validation of HSiKAN joint mix (cycle + walk slots in
# the same alpha mixer) vs paired cycle-only baseline on Bitcoin Alpha
# and Bitcoin OTC.  Plus strict-protocol re-run to test endpoint
# sigma-leakage cost.
#
# Single-seed today: joint mix BA 0.979 (vs cycle 0.939), OTC 0.983
# (vs SiGAT 0.934).  This script locks the win to 5-seed paired-Delta
# at iso-param (h=16; both joint and cycle land at the same param
# count since node-embedding dominates for these graph sizes).
#
# Generated 2026-05-08.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl"
LOG_DIR="/tmp/joint_overnight_2026_05_08"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"   # truncate

echo "[overnight] $(date +'%Y-%m-%d %H:%M:%S') start" | tee -a "$LOG_DIR/orchestrator.log"

# 1. Wait for any in-flight run_final_cell to free the GPU.
echo "[overnight] waiting for GPU to free up..." | tee -a "$LOG_DIR/orchestrator.log"
while pgrep -af "python -m signedkan_wip.src.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 30
done
echo "[overnight] GPU is free, starting runs" | tee -a "$LOG_DIR/orchestrator.log"

run_cell() {
    local label="$1"; shift
    local seed="$1"; shift
    local dataset="$1"; shift
    # Remaining args are env-var "K=V" tokens.
    local logf="$LOG_DIR/${label}_seed${seed}.log"
    local t0=$(date +%s)
    echo "[overnight] $(date +%H:%M:%S) START $label seed=$seed dataset=$dataset" \
        | tee -a "$LOG_DIR/orchestrator.log"
    env "$@" python -m signedkan_wip.src.run_final_cell \
        --dataset "$dataset" --hidden 16 --n-epochs 80 --seed "$seed" \
        > "$logf" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        # Annotate with run_label, then append to JSONL.
        echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = '$label'
d['elapsed_s'] = $elapsed
print(json.dumps(d))
" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(f"{json.loads(sys.stdin.read())[\"auc\"]:.4f}")')
        echo "[overnight] $(date +%H:%M:%S) OK    $label/$seed AUC=$auc  (${elapsed}s)" \
            | tee -a "$LOG_DIR/orchestrator.log"
    else
        echo "[overnight] $(date +%H:%M:%S) FAIL  $label/$seed (rc=$rc, see $logf)" \
            | tee -a "$LOG_DIR/orchestrator.log"
    fi
}

# Phase A: paired joint-mix vs cycle-only baseline (default protocol).
for seed in 0 1 2 3 4; do
    run_cell "joint_ba"  "$seed" "bitcoin_alpha" \
        "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" "HSIKAN_CYCLE_BATCH=4000"
    run_cell "cycle_ba"  "$seed" "bitcoin_alpha"
    run_cell "joint_otc" "$seed" "bitcoin_otc" \
        "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" "HSIKAN_CYCLE_BATCH=4000"
    run_cell "cycle_otc" "$seed" "bitcoin_otc"
done

# Phase B: joint mix under strict protocol (test endpoint sigma-leakage).
for seed in 0 1 2 3 4; do
    run_cell "joint_ba_strict"  "$seed" "bitcoin_alpha" \
        "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" "HSIKAN_CYCLE_BATCH=4000" \
        "HSIKAN_STRICT_PROTOCOL=1"
    run_cell "joint_otc_strict" "$seed" "bitcoin_otc" \
        "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" "HSIKAN_CYCLE_BATCH=4000" \
        "HSIKAN_STRICT_PROTOCOL=1"
done

echo "[overnight] $(date +'%Y-%m-%d %H:%M:%S') DONE — results in $RESULTS_FILE" \
    | tee -a "$LOG_DIR/orchestrator.log"

# Quick summary.
python - <<'PY' | tee -a "$LOG_DIR/orchestrator.log"
import json, statistics, pathlib, collections
p = pathlib.Path("signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl")
groups = collections.defaultdict(list)
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    groups[r["run_label"]].append(r["auc"])
print()
print(f"{'run_label':<22} {'n':>3}  {'mean':>7}  {'std':>6}  {'all':>30}")
for k in sorted(groups):
    aucs = groups[k]
    if len(aucs) >= 2:
        m = statistics.mean(aucs); s = statistics.stdev(aucs)
    elif aucs:
        m = aucs[0]; s = 0.0
    else:
        continue
    print(f"{k:<22} {len(aucs):>3}  {m:.4f}  {s:.4f}  "
          f"{','.join(f'{a:.3f}' for a in aucs)}")
PY
