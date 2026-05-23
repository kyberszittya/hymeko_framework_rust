#!/bin/bash
# 5-seed validation of the full-mix c2,c3,c4,c5,w2,w3 + Highway-quat
# attention configuration that hit 0.9098 single-seed on Slashdot at
# h=4 (vs paper SGT 0.897 ± .002).  Locks the single-seed claim into
# a 5-seed mean ± std.
#
# Each run ~9-15 min depending on cycle/walk enumeration variance.
# Generated 2026-05-08.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_c5_5seed_2026_05_08.jsonl"
LOG_DIR="/tmp/slashdot_c5_5seed_2026_05_08"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[c5seed] $(date +'%H:%M:%S') start"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/c5full_seed${seed}.log"
    local t0=$(date +%s)
    echo "[c5seed] $(date +%H:%M:%S) START seed=$seed"
    HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
    HSIKAN_ATTENTION_M_E=quaternion \
    HSIKAN_ATTENTION_HIGHWAY=1 \
    HSIKAN_CYCLE_BATCH=2000 \
    HSIKAN_MAX_K3=200000 \
    HSIKAN_MAX_K2=200000 \
    python -m signedkan_wip.experiments.runs.run_final_cell \
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
d['run_label'] = 'c5full'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[c5seed] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[c5seed] $(date +%H:%M:%S) FAIL  seed=$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[c5seed] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib
p = pathlib.Path("signedkan_wip/experiments/results/slashdot_c5_5seed_2026_05_08.jsonl")
aucs = []
for line in p.read_text().splitlines():
    if line.strip():
        aucs.append(json.loads(line)["auc"])
print()
if aucs:
    m = statistics.mean(aucs)
    s = statistics.stdev(aucs) if len(aucs) > 1 else 0
    print(f"c5full  n={len(aucs)}  mean={m:.4f}  std={s:.4f}")
    print(f"  per-seed: {','.join(f'{a:.4f}' for a in aucs)}")
    sgt_mean, sgt_std = 0.897, 0.002
    pooled = (s**2 + sgt_std**2) ** 0.5
    delta = m - sgt_mean
    print(f"  vs SGT 0.897 ± .002 → Δ={delta:+.4f}, pooled σ={pooled:.4f}, "
          f"sigma units={delta/pooled if pooled > 0 else 'inf':+.2f}σ")
PY
