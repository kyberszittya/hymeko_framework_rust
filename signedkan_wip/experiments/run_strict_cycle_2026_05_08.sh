#!/bin/bash
# Symmetry check: cycle-only HSiKAN under strict protocol, 5 seeds x
# {BA, OTC}.  Tests whether the AUC=0.500 collapse seen on joint mix
# under strict is protocol (paper's known caveat) or architecture
# (joint mix specific).
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl"
LOG_DIR="/tmp/joint_overnight_2026_05_08"

run_cell() {
    local label="$1"; shift
    local seed="$1"; shift
    local dataset="$1"; shift
    local logf="$LOG_DIR/${label}_seed${seed}.log"
    local t0=$(date +%s)
    echo "[strict] $(date +%H:%M:%S) START $label seed=$seed"
    env "$@" python -m signedkan_wip.src.run_final_cell \
        --dataset "$dataset" --hidden 16 --n-epochs 80 --seed "$seed" \
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
        auc=$(echo "$result" | python -c 'import sys,json;print(f"{json.loads(sys.stdin.read())[\"auc\"]:.4f}")')
        echo "[strict] $(date +%H:%M:%S) OK    $label/$seed AUC=$auc (${elapsed}s)"
    else
        echo "[strict] $(date +%H:%M:%S) FAIL  $label/$seed"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "cycle_ba_strict"  "$seed" "bitcoin_alpha" \
        "HSIKAN_STRICT_PROTOCOL=1"
    run_cell "cycle_otc_strict" "$seed" "bitcoin_otc" \
        "HSIKAN_STRICT_PROTOCOL=1"
done

# Summary
python - <<'PY'
import json, statistics, pathlib, collections
p = pathlib.Path("signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl")
groups = collections.defaultdict(list)
for line in p.read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    groups[r["run_label"]].append(r["auc"])
for k in sorted(groups):
    aucs = groups[k]
    if len(aucs) >= 2:
        m = statistics.mean(aucs); s = statistics.stdev(aucs)
    else:
        m = aucs[0] if aucs else 0; s = 0.0
    print(f"{k:<22} n={len(aucs)}  mean={m:.4f}  std={s:.4f}")
PY
