#!/bin/bash
# 5-seed Epinions push at edge_cr + balance pruner + cycle cache.
#
# Stacks today's three new levers on top of the in-flight edge_cr
# baseline:
#   1. HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr   (Slashdot SOTA gate)
#   2. HSIKAN_TOPK_MODE=per_vertex + PRUNER=balance
#      (+4.6pp on BA quick test; consistent with
#       project_balance_pruner_win_2026_05_05's +4.5-5pp on Bitcoin Alpha)
#   3. HYMEKO_CYCLE_CACHE=1
#      (cycle space is seed-independent → first seed populates,
#       seeds 1-4 hit cache → ~5x speedup on enum)
#   plus HSIKAN_TRITON_KERNEL=1 + BACKWARD=1 (1.37x training speedup).
#
# Hypothesis: balance pruner gives the model axiom-aware cycles
# (Cartwright-Harary structurally-balanced subset) where default
# enumeration was diluting signal with unbalanced cycles.  All
# previous Epinions ablations (project_epinions_ceiling) used
# default enum — the architectural ceiling at 0.84 was diagnosed
# against the WRONG cycle set.
#
# Auto-queues behind any in-flight run_final_cell process.
# Generated 2026-05-10.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/epinions_balance_5seed_2026_05_10.jsonl"
LOG_DIR="/tmp/epinions_balance_5seed_2026_05_10"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[ep_bal] $(date +'%Y-%m-%d %H:%M:%S') start"

echo "[ep_bal] waiting for GPU..."
while pgrep -af "python -m signedkan_wip.experiments.runs.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[ep_bal] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/balance_seed${seed}.log"
    local t0=$(date +%s)
    echo "[ep_bal] $(date +%H:%M:%S) START seed=$seed"
    HYMEKO_CYCLE_CACHE=1 \
    HSIKAN_TOPK_MODE=per_vertex \
    HSIKAN_TOPK_K=128 \
    HSIKAN_TOPK_SCORER=fraction_negative \
    HSIKAN_TOPK_PRUNER=balance \
    HSIKAN_TRITON_KERNEL=1 \
    HSIKAN_TRITON_BACKWARD=1 \
    HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
    HSIKAN_ATTENTION_M_E=quaternion \
    HSIKAN_ATTENTION_HIGHWAY=1 \
    HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
    HSIKAN_CYCLE_BATCH=2000 \
    HSIKAN_MAX_K3=200000 \
    HSIKAN_MAX_K2=200000 \
    python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed "$seed" \
        > "$logf" 2>&1
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = 'edge_cr_balance'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[ep_bal] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[ep_bal] $(date +%H:%M:%S) FAIL  seed=$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[ep_bal] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib
new = pathlib.Path("signedkan_wip/experiments/results/epinions_balance_5seed_2026_05_10.jsonl")
ecr = pathlib.Path("signedkan_wip/experiments/results/epinions_edge_cr_5seed_2026_05_09.jsonl")
old = pathlib.Path("signedkan_wip/experiments/results/epinions_overnight_2026_05_09.jsonl")

def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

new_rows = load(new); ecr_rows = load(ecr); old_rows = load(old)

print()
def report(label, rows):
    if not rows: return
    aucs = [r["auc"] for r in rows]
    walls = [r.get("elapsed_s", 0) for r in rows]
    am = statistics.mean(aucs); asd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    em = statistics.mean(walls)
    print(f"{label:<32} n={len(aucs)}  AUC={am:.4f}+/-{asd:.4f}  "
          f"wall={em:.0f}s/seed  per-seed: {','.join(f'{a:.4f}' for a in aucs)}")

big = [r for r in old_rows if r["run_label"] == "bigger_caps"]
if big:
    print(f"{'bigger_caps (1-seed)':<32} n=1  AUC={big[0]['auc']:.4f}  wall={big[0]['elapsed_s']:.0f}s")
report("edge_cr (5-seed)", ecr_rows)
report("edge_cr + balance (5-seed)", new_rows)

if len(ecr_rows) == len(new_rows) == 5:
    da = [n["auc"] - e["auc"] for n, e in zip(new_rows, ecr_rows)]
    md = statistics.mean(da); sd = statistics.stdev(da) if len(da) > 1 else 0.0
    sigma = md * (len(da) ** 0.5) / max(sd, 1e-9)
    print()
    print(f"  Paired delta (balance vs no-balance): {md:+.4f} +/- {sd:.4f}  sigma={sigma:+.2f}")
print()
print("SGT Epinions reference: 0.941 (target gap to close)")
PY
