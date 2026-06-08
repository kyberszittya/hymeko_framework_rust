#!/bin/bash
# 5-seed Epinions edge_cr push, kernel-enabled.
#
# Tests whether the per-edge Catmull-Rom Highway gate
# (HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr — Slashdot SOTA winner at
# +3.06 sigma paired) opens where the scalar sigmoid gate never
# did on Epinions.  Per project_epinions_ceiling_2026_05_09:
# "Highway gates stayed at sigmoid(-3) ~ 0.05 in EVERY variant.
# Attention has not once opened on Epinions."  But every variant
# tested used the SCALAR gate.  edge_cr is structurally different
# — a learnable Catmull-Rom curve over the query embedding.
#
# Recipe = bigger_caps (best Epinions single-seed at 0.8409) with:
#   + HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr   (new gate kind)
#   + HSIKAN_TRITON_KERNEL=1 + BACKWARD=1     (1.37x speedup)
#
# Baseline: bigger_caps single-seed AUC=0.8409 (2026-05-09 morning).
# Compare against SGT 0.941 — current gap -0.10.
#
# Wall: 5 seeds * ~85 min/seed (kernel-ON) = ~7 hours overnight.
#
# Generated 2026-05-09.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/epinions_edge_cr_5seed_2026_05_09.jsonl"
LOG_DIR="/tmp/epinions_edge_cr_5seed_2026_05_09"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[ep_ecr] $(date +'%Y-%m-%d %H:%M:%S') start"

echo "[ep_ecr] waiting for GPU..."
while pgrep -af "python -m signedkan_wip.experiments.runs.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[ep_ecr] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/edge_cr_seed${seed}.log"
    local t0=$(date +%s)
    echo "[ep_ecr] $(date +%H:%M:%S) START seed=$seed"
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
d['run_label'] = 'edge_cr'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc gate
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        gate=$(echo "$result" | python -c 'import sys,json; r=json.loads(sys.stdin.read()); print(r.get("attn_gate_mean", "?"))')
        echo "[ep_ecr] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[ep_ecr] $(date +%H:%M:%S) FAIL  seed=$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[ep_ecr] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib
new = pathlib.Path("signedkan_wip/experiments/results/epinions_edge_cr_5seed_2026_05_09.jsonl")
old = pathlib.Path("signedkan_wip/experiments/results/epinions_overnight_2026_05_09.jsonl")

def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

new_rows = load(new)
old_rows = load(old)

print()
def report(label, rows):
    if not rows: return
    aucs = [r["auc"] for r in rows]
    walls = [r.get("elapsed_s", 0) for r in rows]
    am = statistics.mean(aucs)
    asd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    em = statistics.mean(walls)
    per_seed = ",".join(f"{a:.4f}" for a in aucs)
    print(f"{label:<28} n={len(aucs)}  AUC={am:.4f}+/-{asd:.4f}  "
          f"wall={em:.0f}s/seed  per-seed: {per_seed}")

# bigger_caps was a single seed; report that as the reference.
big = [r for r in old_rows if r["run_label"] == "bigger_caps"]
if big:
    print(f"{'bigger_caps (1-seed ref)':<28} n=1  AUC={big[0]['auc']:.4f}  wall={big[0]['elapsed_s']:.0f}s")
report("edge_cr 5-seed (today)", new_rows)

print()
print("SGT Epinions reference: 0.941 (gap to close)")
PY
