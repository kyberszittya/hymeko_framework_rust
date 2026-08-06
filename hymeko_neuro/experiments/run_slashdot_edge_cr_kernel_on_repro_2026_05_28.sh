#!/bin/bash
# Reproduction sanity-check of the 2026-05-09 Slashdot edge_cr kernel-ON
# SOTA (mean AUC 0.9070 ± .0029, 5-seed) under today's code + torch.
# Preserves the May-9 result file untouched; writes today's 5-seed run
# to a new dated jsonl and compares the two at the end.
#
# Recipe is bit-identical to run_slashdot_edge_cr_kernel_on_2026_05_09.sh
# lines 41-53. Generated 2026-05-28.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OLD_FILE="hymeko_neuro/experiments/results/slashdot_edge_cr_kernel_on_2026_05_09.jsonl"
NEW_FILE="hymeko_neuro/experiments/results/slashdot_edge_cr_kernel_on_2026_05_28_repro.jsonl"
LOG_DIR="/tmp/slashdot_edge_cr_kernel_on_repro_2026_05_28"
mkdir -p "$LOG_DIR"
> "$NEW_FILE"

echo "[repro] $(date -Is) start; preserving $OLD_FILE; writing $NEW_FILE"

# Built-in GPU politeness: wait if any run_final_cell is alive.
echo "[repro] waiting for GPU..."
while pgrep -af "python -m hymeko_neuro.experiments.runs.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[repro] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/edge_cr_kernel_on_repro_seed${seed}.log"
    local t0; t0=$(date +%s)
    echo "[repro] $(date +%H:%M:%S) START seed=$seed"
    HSIKAN_TRITON_KERNEL=1 \
    HSIKAN_TRITON_BACKWARD=1 \
    HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
    HSIKAN_ATTENTION_M_E=quaternion \
    HSIKAN_ATTENTION_HIGHWAY=1 \
    HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
    HSIKAN_CYCLE_BATCH=2000 \
    HSIKAN_MAX_K3=200000 \
    HSIKAN_MAX_K2=200000 \
    /home/kyberszittya/miniconda3/bin/python -m hymeko_neuro.experiments.runs.run_final_cell \
        --dataset slashdot --hidden 4 --n-epochs 80 \
        --max-k4 200000 --seed "$seed" \
        > "$logf" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$result" ]; then
        echo "$result" | /home/kyberszittya/miniconda3/bin/python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = 'edge_cr_kernel_on_repro_2026_05_28'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$NEW_FILE"
        local auc
        auc=$(echo "$result" | /home/kyberszittya/miniconda3/bin/python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[repro] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[repro] $(date +%H:%M:%S) FAIL  seed=$seed rc=$rc (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[repro] $(date +%H:%M:%S) all seeds done"

/home/kyberszittya/miniconda3/bin/python - <<PY
import json, statistics, pathlib

old = pathlib.Path("$OLD_FILE")
new = pathlib.Path("$NEW_FILE")

def load(p):
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

old_rows = load(old)
new_rows = load(new)

def report(label, rows):
    if not rows: print(f"{label}: (no rows)"); return
    aucs = [r["auc"] for r in rows]
    elapsed = [r.get("elapsed_s", 0) for r in rows]
    am = statistics.mean(aucs)
    asd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    em = statistics.mean(elapsed)
    print(f"{label:<40} n={len(aucs)}  AUC={am:.4f}+/-{asd:.4f}  "
          f"wall={em:.0f}s/seed  per-seed: "
          f"{','.join(f'{a:.4f}' for a in aucs)}")

print()
report("May-09 kernel-ON (preserved)", old_rows)
report("May-28 kernel-ON (repro)",     new_rows)

if len(old_rows) == len(new_rows) and len(new_rows) > 1:
    o = [r["auc"] for r in old_rows]
    n = [r["auc"] for r in new_rows]
    deltas = [a - b for a, b in zip(n, o)]
    md = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    sigma = md * (len(deltas) ** 0.5) / max(sd, 1e-9)
    print()
    print(f"  Paired delta (May-28 minus May-09): {md:+.4f} +/- {sd:.4f}  sigma={sigma:+.2f}")
    print(f"  Verdict: {'reproduces (within seed noise)' if abs(sigma) < 2 else 'DRIFTED'}")
PY
echo "[repro] $(date -Is) DONE"
