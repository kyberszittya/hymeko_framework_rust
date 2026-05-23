#!/bin/bash
# 5-seed Slashdot validation of joint+Highway-quat at h=4 (paper Pareto).
#
# Three conditions × 5 seeds = 15 runs:
#   1. cycle_h4     — cycle-only K=(3,4,5), h=4 (anchors against paper 0.861)
#   2. joint_h4     — HSIKAN_MIXED_TUPLES=c3,c4,w2,w3, h=4 (paired baseline
#                     measuring walk contribution alone)
#   3. joint_attn_h4 — joint + Highway-quat attention (the winning config,
#                     0.8924 single-seed)
#
# Goal: convert the single-seed 0.8924 into a 5-seed mean ± std for paper
# claim against SGT 0.897 ± .002.  Cycle-batch=2000, max_k4=max_k3=200K
# match the single-seed protocol that produced the 0.8924 number.
#
# Generated 2026-05-08.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_5seed_2026_05_08.jsonl"
LOG_DIR="/tmp/slashdot_5seed_2026_05_08"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[5seed] $(date +'%H:%M:%S') start"

run_cell() {
    local label="$1"; shift
    local seed="$1"; shift
    local logf="$LOG_DIR/${label}_seed${seed}.log"
    local t0=$(date +%s)
    echo "[5seed] $(date +%H:%M:%S) START $label/$seed"
    env "$@" python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset slashdot --hidden 4 --n-epochs 80 \
        --max-k4 200000 --seed "$seed" \
        > "$logf" 2>&1
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = '$label'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[5seed] $(date +%H:%M:%S) OK    $label/$seed AUC=$auc (${elapsed}s)"
    else
        echo "[5seed] $(date +%H:%M:%S) FAIL  $label/$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "cycle_h4" "$seed"
    run_cell "joint_h4" "$seed" \
        "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" \
        "HSIKAN_CYCLE_BATCH=2000" \
        "HSIKAN_MAX_K3=200000"
    run_cell "joint_attn_h4" "$seed" \
        "HSIKAN_MIXED_TUPLES=c3,c4,w2,w3" \
        "HSIKAN_CYCLE_BATCH=2000" \
        "HSIKAN_MAX_K3=200000" \
        "HSIKAN_ATTENTION_M_E=quaternion" \
        "HSIKAN_ATTENTION_HIGHWAY=1"
done

echo "[5seed] $(date +'%H:%M:%S') DONE — results in $RESULTS_FILE"

# Summary
python - <<'PY'
import json, statistics, pathlib, collections
p = pathlib.Path("signedkan_wip/experiments/results/slashdot_5seed_2026_05_08.jsonl")
groups = collections.defaultdict(list)
for line in p.read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    groups[r["run_label"]].append(r["auc"])
print()
print(f"{'label':<22} {'n':>3}  {'mean':>7}  {'std':>6}  {'all'}")
for k in sorted(groups):
    aucs = groups[k]
    if len(aucs) >= 2:
        m = statistics.mean(aucs); s = statistics.stdev(aucs)
    else:
        m = aucs[0] if aucs else 0; s = 0.0
    aucs_str = ",".join(f"{a:.3f}" for a in aucs)
    print(f"{k:<22} {len(aucs):>3}  {m:.4f}  {s:.4f}  {aucs_str}")
PY
