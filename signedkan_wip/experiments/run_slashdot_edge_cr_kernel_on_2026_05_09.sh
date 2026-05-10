#!/bin/bash
# 5-seed Slashdot edge_cr run with HSIKAN_TRITON_KERNEL=1.
#
# Goal: quantify the training-time wall-clock speedup the Triton
# fused inner forward+backward kernel buys on the validated SOTA
# recipe (edge_cr Highway gate, mixed cycles + walks, h=4).
#
# Baseline (kernel OFF, 2026-05-09 morning):
#   slashdot_edge_cr_5seed_2026_05_09.jsonl
#   mean AUC 0.9067 ± .0034; mean wall ~1084 s/seed
#
# Hypothesis: kernel ON should produce identical AUC distribution
# (within seed noise) at lower per-seed wall time.
#
# Generated 2026-05-09.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_edge_cr_kernel_on_2026_05_09.jsonl"
LOG_DIR="/tmp/slashdot_edge_cr_kernel_on_2026_05_09"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[edge_cr_kernel_on] $(date +'%Y-%m-%d %H:%M:%S') start"

echo "[edge_cr_kernel_on] waiting for GPU..."
while pgrep -af "python -m signedkan_wip.src.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[edge_cr_kernel_on] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/edge_cr_kernel_on_seed${seed}.log"
    local t0=$(date +%s)
    echo "[edge_cr_kernel_on] $(date +%H:%M:%S) START seed=$seed"
    HSIKAN_TRITON_KERNEL=1 \
    HSIKAN_TRITON_BACKWARD=1 \
    HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
    HSIKAN_ATTENTION_M_E=quaternion \
    HSIKAN_ATTENTION_HIGHWAY=1 \
    HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
    HSIKAN_CYCLE_BATCH=2000 \
    HSIKAN_MAX_K3=200000 \
    HSIKAN_MAX_K2=200000 \
    python -m signedkan_wip.src.run_final_cell \
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
d['run_label'] = 'edge_cr_kernel_on'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[edge_cr_kernel_on] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[edge_cr_kernel_on] $(date +%H:%M:%S) FAIL  seed=$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[edge_cr_kernel_on] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib

base = pathlib.Path("signedkan_wip/experiments/results/slashdot_edge_cr_5seed_2026_05_09.jsonl")
new  = pathlib.Path("signedkan_wip/experiments/results/slashdot_edge_cr_kernel_on_2026_05_09.jsonl")

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
    print(f"{label:<32} n={len(aucs)}  AUC={am:.4f}±{asd:.4f}  "
          f"wall={em:.0f}s/seed  per-seed: "
          f"{','.join(f'{a:.4f}' for a in aucs)}")

report("edge_cr kernel OFF (baseline)", base_rows)
report("edge_cr kernel ON  (today)",    new_rows)

if len(base_rows) == len(new_rows):
    base_aucs = [r["auc"] for r in base_rows]
    new_aucs  = [r["auc"] for r in new_rows]
    base_walls = [r["elapsed_s"] for r in base_rows]
    new_walls  = [r["elapsed_s"] for r in new_rows]

    deltas_auc = [n - b for n, b in zip(new_aucs, base_aucs)]
    md = statistics.mean(deltas_auc)
    sd = statistics.stdev(deltas_auc) if len(deltas_auc) > 1 else 0.0
    sigma = md * (len(deltas_auc) ** 0.5) / max(sd, 1e-9)

    walls_speedup = [b / n for b, n in zip(base_walls, new_walls)]
    msp = statistics.mean(walls_speedup)
    print()
    print(f"  AUC paired delta (kernel-ON minus OFF): {md:+.4f} +/- {sd:.4f}  sigma={sigma:+.2f}")
    print(f"  Wall-time speedup (per seed):         {msp:.2f}x  "
          f"(per-seed: {','.join(f'{x:.2f}' for x in walls_speedup)})")
    print(f"  Total wall: kernel-OFF={sum(base_walls)}s, kernel-ON={sum(new_walls)}s, "
          f"saved {sum(base_walls)-sum(new_walls)}s ({(1 - sum(new_walls)/sum(base_walls))*100:.0f}%)")
PY
