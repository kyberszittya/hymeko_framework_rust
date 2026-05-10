#!/bin/bash
# Overnight Epinions exploration — find a config that lifts above
# today's 0.8053 single-seed baseline (h=4, c2,c3,c4,c5,w2,w3,
# max_k4=100K).  SGT 0.941 is the SOTA target; SGCN 0.928 is the
# walk-rich middle ground.
#
# Diagnostic from today: ALL attn_gate values stayed at sigmoid(-3) ≈
# 0.05.  Attention found nothing useful.  Hypothesis: per-edge
# softmax is too dispersed (Epinions has more cycles per edge than
# Slashdot), so per-edge attention pool ≈ uniform pool, gate
# gradient near zero, gate stays at init.
#
# Six single-seed variants, each attacking a different bottleneck:
#   1. bigger_caps     — more tuples per arity slot (200K vs 100K)
#   2. longer_walks    — add w4, w5; longer-range walk signal
#   3. h_8             — wider model; helps if h=4 was too narrow
#   4. h_16_smaller_K  — paper's reference hidden, with safer caps
#   5. aux_a1_a2       — Phase A entropy regularisers on top of recipe
#   6. direct_msg      — SGCN-style sign-conditional propagation path
#
# Each run ~30-60 min depending on graph + slot count + cap.  Runs
# sequentially.  Waits for the GPU to free up first.
#
# Generated 2026-05-09.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/epinions_overnight_2026_05_09.jsonl"
LOG_DIR="/tmp/epinions_overnight_2026_05_09"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[ep] $(date +'%Y-%m-%d %H:%M:%S') start"

# Wait for any other run_final_cell to finish first.
echo "[ep] waiting for GPU to free..."
while pgrep -f "python -m signedkan_wip.src.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[ep] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local label="$1"; shift
    local logf="$LOG_DIR/${label}.log"
    local t0=$(date +%s)
    echo "[ep] $(date +%H:%M:%S) START $label"
    env "$@" python -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 0 \
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
        local auc gates alphas
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        gates=$(echo "$result" | python -c 'import sys,json;d=json.loads(sys.stdin.read());g=d.get("attn_gate") or [];print("[" + ",".join(f"{x:.2f}" for x in g) + "]" if g else "none")')
        echo "[ep] $(date +%H:%M:%S) OK    $label AUC=$auc  gates=$gates  (${elapsed}s)"
    else
        echo "[ep] $(date +%H:%M:%S) FAIL  $label"
    fi
}

BASE_RECIPE=("HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3"
             "HSIKAN_ATTENTION_M_E=quaternion"
             "HSIKAN_ATTENTION_HIGHWAY=1"
             "HSIKAN_CYCLE_BATCH=2000"
             "HSIKAN_MAX_K3=100000"
             "HSIKAN_MAX_K2=100000")

# 1. Bigger caps
run_cell "bigger_caps" \
    "HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3" \
    "HSIKAN_ATTENTION_M_E=quaternion" \
    "HSIKAN_ATTENTION_HIGHWAY=1" \
    "HSIKAN_CYCLE_BATCH=2000" \
    "HSIKAN_MAX_K3=200000" \
    "HSIKAN_MAX_K2=200000"

# 2. Longer walks (add w4, w5)
run_cell "longer_walks" \
    "HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3,w4,w5" \
    "HSIKAN_ATTENTION_M_E=quaternion" \
    "HSIKAN_ATTENTION_HIGHWAY=1" \
    "HSIKAN_CYCLE_BATCH=2000" \
    "HSIKAN_MAX_K3=100000" \
    "HSIKAN_MAX_K2=100000"

# 3. Wider hidden (h=8). Need separate launch with --hidden 8.
echo "[ep] $(date +%H:%M:%S) START h_8"
t0=$(date +%s)
HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
HSIKAN_ATTENTION_M_E=quaternion \
HSIKAN_ATTENTION_HIGHWAY=1 \
HSIKAN_CYCLE_BATCH=2000 \
HSIKAN_MAX_K3=100000 \
HSIKAN_MAX_K2=100000 \
python -m signedkan_wip.src.run_final_cell \
    --dataset epinions --hidden 8 --n-epochs 80 \
    --max-k4 100000 --seed 0 \
    > "$LOG_DIR/h_8.log" 2>&1
elapsed=$(( $(date +%s) - t0 ))
result=$(grep -E '^\{"dataset"' "$LOG_DIR/h_8.log" | tail -1)
if [ -n "$result" ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = 'h_8'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
    auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
    echo "[ep] $(date +%H:%M:%S) OK    h_8 AUC=$auc (${elapsed}s)"
else
    echo "[ep] $(date +%H:%M:%S) FAIL  h_8"
fi

# 4. h=16 with smaller K cap (memory safety)
echo "[ep] $(date +%H:%M:%S) START h_16_small"
t0=$(date +%s)
HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
HSIKAN_ATTENTION_M_E=quaternion \
HSIKAN_ATTENTION_HIGHWAY=1 \
HSIKAN_CYCLE_BATCH=2000 \
HSIKAN_MAX_K3=50000 \
HSIKAN_MAX_K2=50000 \
python -m signedkan_wip.src.run_final_cell \
    --dataset epinions --hidden 16 --n-epochs 80 \
    --max-k4 50000 --seed 0 \
    > "$LOG_DIR/h_16_small.log" 2>&1
elapsed=$(( $(date +%s) - t0 ))
result=$(grep -E '^\{"dataset"' "$LOG_DIR/h_16_small.log" | tail -1)
if [ -n "$result" ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = 'h_16_small'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
    auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
    echo "[ep] $(date +%H:%M:%S) OK    h_16_small AUC=$auc (${elapsed}s)"
else
    echo "[ep] $(date +%H:%M:%S) FAIL  h_16_small"
fi

# 5. Phase A1+A2 aux entropy on top of base recipe
run_cell "aux_a1_a2" \
    "${BASE_RECIPE[@]}" \
    "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.01" \
    "HSIKAN_ATTN_ENTROPY_LAMBDA=0.01"

# 6. Direct messaging (SGCN-style path)
run_cell "direct_msg" \
    "${BASE_RECIPE[@]}" \
    "HSIKAN_DIRECT_MESSAGING=1"

echo "[ep] $(date +'%H:%M:%S') DONE — results in $RESULTS_FILE"

python - <<'PY'
import json, statistics, pathlib
p = pathlib.Path("signedkan_wip/experiments/results/epinions_overnight_2026_05_09.jsonl")
print()
print(f"{'label':<16}{'AUC':>9}  {'params':>9}  {'wall (s)':>10}  alpha gates")
print(f"{'baseline':<16}{0.8053:9.4f}  {527610:>9}  {'~1.5h':>10}  (today's c5full)")
for line in p.read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    g = r.get("attn_gate") or []
    g_str = "[" + ",".join(f"{x:.2f}" for x in g) + "]" if g else "none"
    print(f"{r['run_label']:<16}{r['auc']:9.4f}  {r['n_params']:>9}  "
          f"{r.get('elapsed_s', '?'):>10}  {g_str}")
PY
