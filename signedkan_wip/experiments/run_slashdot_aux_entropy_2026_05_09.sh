#!/bin/bash
# Phase A1 + A2 sweep on Slashdot, applied on top of the validated
# c5full + Highway-quat baseline (5-seed mean = 0.9035 ± .0044).
#
# Conditions (5 seeds each, 10 runs total):
#   1. aux_a1     — HSIKAN_ALPHA_ENTROPY_LAMBDA=0.01
#   2. aux_a1_a2  — HSIKAN_ALPHA_ENTROPY_LAMBDA=0.01 + HSIKAN_ATTN_ENTROPY_LAMBDA=0.01
#
# Compared against the existing c5full 5-seed
# (slashdot_c5_5seed_2026_05_08.jsonl).  Acceptance: paired Δ > 1σ on
# at least one condition.
#
# Generated 2026-05-09.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_aux_entropy_2026_05_09.jsonl"
LOG_DIR="/tmp/slashdot_aux_entropy_2026_05_09"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[aux] $(date +'%H:%M:%S') start"

run_cell() {
    local label="$1"; shift
    local seed="$1"; shift
    local logf="$LOG_DIR/${label}_seed${seed}.log"
    local t0=$(date +%s)
    echo "[aux] $(date +%H:%M:%S) START $label/$seed"
    env "$@" python -m signedkan_wip.src.run_final_cell \
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
        echo "[aux] $(date +%H:%M:%S) OK    $label/$seed AUC=$auc (${elapsed}s)"
    else
        echo "[aux] $(date +%H:%M:%S) FAIL  $label/$seed"
    fi
}

BASE=("HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3"
      "HSIKAN_ATTENTION_M_E=quaternion"
      "HSIKAN_ATTENTION_HIGHWAY=1"
      "HSIKAN_CYCLE_BATCH=2000"
      "HSIKAN_MAX_K3=200000"
      "HSIKAN_MAX_K2=200000")

for seed in 0 1 2 3 4; do
    run_cell "aux_a1"    "$seed" "${BASE[@]}" \
        "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.01"
    run_cell "aux_a1_a2" "$seed" "${BASE[@]}" \
        "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.01" \
        "HSIKAN_ATTN_ENTROPY_LAMBDA=0.01"
done

echo "[aux] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib, collections
new = pathlib.Path("signedkan_wip/experiments/results/slashdot_aux_entropy_2026_05_09.jsonl")
old = pathlib.Path("signedkan_wip/experiments/results/slashdot_c5_5seed_2026_05_08.jsonl")
groups = collections.defaultdict(list)
for line in new.read_text().splitlines():
    if line.strip():
        r = json.loads(line); groups[r["run_label"]].append(r["auc"])
# Reuse the c5full baseline for paired Δ.
base_aucs = []
if old.exists():
    for line in old.read_text().splitlines():
        if line.strip():
            base_aucs.append(json.loads(line)["auc"])
print()
print(f"{'label':<14}{'n':>3}  {'mean':>7}  {'std':>6}  per-seed")
print(f"{'baseline':<14}{len(base_aucs):>3}  "
      f"{statistics.mean(base_aucs):.4f}  "
      f"{statistics.stdev(base_aucs):.4f}  "
      f"{','.join(f'{a:.4f}' for a in base_aucs)}")
for k in sorted(groups):
    aucs = groups[k]
    if len(aucs) >= 2:
        m = statistics.mean(aucs); s = statistics.stdev(aucs)
    else:
        m = aucs[0] if aucs else 0; s = 0.0
    print(f"{k:<14}{len(aucs):>3}  {m:.4f}  {s:.4f}  "
          f"{','.join(f'{a:.4f}' for a in aucs)}")
    # Paired Δ vs baseline if seeds align (assume seeds 0..4 in order)
    if len(aucs) == len(base_aucs):
        deltas = [a - b for a, b in zip(aucs, base_aucs)]
        m_d = statistics.mean(deltas)
        s_d = statistics.stdev(deltas) if len(deltas) > 1 else 0
        sigma = m_d * (len(deltas) ** 0.5) / max(s_d, 1e-9)
        print(f"{'  paired Δ':<14}{'':>3}  {m_d:+.4f}  {s_d:.4f}  "
              f"{sigma:+.2f}σ")
PY
