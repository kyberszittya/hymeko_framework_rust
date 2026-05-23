#!/bin/bash
# Slashdot SOTA push — 4 variants, single-seed, sequential.
# Goal: lift HSiKAN on Slashdot above the c3,c4,w2,w3 1-seed result
# of 0.864 toward SGT's 0.897.  Memory caps are conservative
# (max_k3=max_k4=100K, cycle_batch=2000) to avoid the 2.3-hr
# expanded-run timeout.
#
# Generated 2026-05-08.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_push_2026_05_08.jsonl"
LOG_DIR="/tmp/slashdot_push_2026_05_08"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

run_cell() {
    local label="$1"; shift
    local logf="$LOG_DIR/${label}.log"
    local t0=$(date +%s)
    echo "[push] $(date +%H:%M:%S) START $label"
    env "$@" python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset slashdot --hidden 16 --n-epochs 80 --seed 0 \
        --max-k4 100000 \
        > "$logf" 2>&1
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        echo "$result" \
          | python -c "import sys, json; d=json.loads(sys.stdin.read()); d['run_label']='$label'; d['elapsed_s']=$elapsed; print(json.dumps(d))" \
          >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json; print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[push] $(date +%H:%M:%S) OK    $label AUC=$auc  (${elapsed}s)"
    else
        echo "[push] $(date +%H:%M:%S) FAIL  $label  (see $logf)"
    fi
}

# 1. Per-edge gate + Gumbel-hard (joint mix recipe + tuple selection)
run_cell "v1_gate_gumbel" \
    "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" \
    "HSIKAN_CYCLE_BATCH=2000" \
    "HSIKAN_MAX_K3=100000" \
    "HSIKAN_PER_EDGE_GATE=1" \
    "HSIKAN_GUMBEL_HARD=1"

# 2. Walks-only with 4 lengths (long walks on dense graph)
run_cell "v2_walks4_only" \
    "HSIKAN_MIXED_TUPLES=w2,w3,w4,w5" \
    "HSIKAN_CYCLE_BATCH=2000" \
    "HSIKAN_MAX_K3=100000"

# 3. Paper's strong-cycle K (k=4,5) + long walks
run_cell "v3_paperK_longwalks" \
    "HSIKAN_MIXED_TUPLES=c4,c5,w3,w4,w5" \
    "HSIKAN_CYCLE_BATCH=2000" \
    "HSIKAN_MAX_K3=100000"

# 4. Maximum slot diversity (4 cycles + 4 walks would be 8 slots — skip;
#    pick 6: c3,c4 + w2..w5)
run_cell "v4_full6" \
    "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3,w4,w5" \
    "HSIKAN_CYCLE_BATCH=2000" \
    "HSIKAN_MAX_K3=100000"

# Summary
python - <<'PY'
import json, pathlib
p = pathlib.Path("signedkan_wip/experiments/results/slashdot_push_2026_05_08.jsonl")
print("\nSlashdot push summary (single seed, baseline 0.864):")
print(f"{'label':<22} {'AUC':>7}  {'alpha (mean)':<60}  {'wall-s'}")
for line in p.read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    label = r.get("run_label","?")
    auc = r.get("auc", float("nan"))
    alphas = r.get("alpha", [])
    labels = r.get("tuple_labels", [])
    pairs = ", ".join(f"{l}={a:.2f}" for l, a in zip(labels, alphas))
    print(f"{label:<22} {auc:.4f}  {pairs:<60}  {r.get('elapsed_s', '?')}")
PY

echo "[push] $(date +%H:%M:%S) DONE — results in $RESULTS_FILE"
