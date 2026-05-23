#!/bin/bash
# 5-seed Slashdot d=16 joint-mix ablation with HSIKAN_TRITON_KERNEL=1.
#
# Goal: test whether wider hidden dim (d=16 vs d=4) helps the validated
# SOTA recipe.  Previously OOM-bound on 8 GiB GPU (~3 GiB peak in
# PyTorch path); now runs in ~200 MB thanks to the fused backward
# kernel (90.6%-93.6% memory savings, 1.40-1.90x tl.dot speedup at
# d>=16 in the gate matvec).
#
# Recipe: SOTA edge_cr + mixed cycles + walks, varying only hidden 4 -> 16.
# Baseline: d=4 mean 0.9067 +/- .0034 (5-seed paired, 2026-05-09 morning).
#
# Auto-queues behind any in-flight run_final_cell process via pgrep loop.
# Generated 2026-05-09.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_d16_joint_mix_2026_05_09.jsonl"
LOG_DIR="/tmp/slashdot_d16_joint_mix_2026_05_09"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[d16] $(date +'%Y-%m-%d %H:%M:%S') start"

echo "[d16] waiting for GPU (will queue if another run_final_cell is in-flight)..."
while pgrep -af "python -m signedkan_wip.experiments.runs.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[d16] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/d16_seed${seed}.log"
    local t0=$(date +%s)
    echo "[d16] $(date +%H:%M:%S) START seed=$seed"
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
        --dataset slashdot --hidden 16 --n-epochs 80 \
        --max-k4 200000 --seed "$seed" \
        > "$logf" 2>&1
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = 'd16_joint_mix'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[d16] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[d16] $(date +%H:%M:%S) FAIL  seed=$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[d16] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib

base = pathlib.Path("signedkan_wip/experiments/results/slashdot_edge_cr_5seed_2026_05_09.jsonl")
new  = pathlib.Path("signedkan_wip/experiments/results/slashdot_d16_joint_mix_2026_05_09.jsonl")

def load(p):
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

base_rows = load(base)
new_rows  = load(new)

print()
def report(label, rows):
    if not rows: return
    aucs    = [r["auc"] for r in rows]
    elapsed = [r.get("elapsed_s", 0) for r in rows]
    am = statistics.mean(aucs)
    asd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    em = statistics.mean(elapsed)
    print(f"{label:<32} n={len(aucs)}  AUC={am:.4f}+/-{asd:.4f}  "
          f"wall={em:.0f}s/seed  per-seed: "
          f"{','.join(f'{a:.4f}' for a in aucs)}")

report("d=4 (baseline, kernel OFF)", base_rows)
report("d=16 (today, kernel ON)",    new_rows)

if len(base_rows) == len(new_rows):
    base_aucs = [r["auc"] for r in base_rows]
    new_aucs  = [r["auc"] for r in new_rows]
    deltas = [n - b for n, b in zip(new_aucs, base_aucs)]
    md = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    sigma = md * (len(deltas) ** 0.5) / max(sd, 1e-9)
    print()
    print(f"  Paired delta (d=16 minus d=4): {md:+.4f} +/- {sd:.4f}  sigma={sigma:+.2f}")
PY
