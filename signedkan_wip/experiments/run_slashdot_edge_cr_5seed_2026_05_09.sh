#!/bin/bash
# 5-seed Slashdot validation of the per-edge KAN-aligned (sigmoid-free)
# Highway gate (`HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr`).
#
# Compared against the c5full + Highway-scalar 5-seed baseline
# (slashdot_c5_5seed_2026_05_08.jsonl mean 0.9035 ± .0044) and the
# A1-aux 5-seed (slashdot_aux_entropy_2026_05_09.jsonl mean
# 0.9050 ± .0050).
#
# Hypothesis: per-edge CR gate adds usable per-edge variation on the
# walk-rich Slashdot regime where attention finds genuine signal,
# unlike the synthetic-toy where per-edge gating regressed.
#
# Generated 2026-05-09.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_FILE="signedkan_wip/experiments/results/slashdot_edge_cr_5seed_2026_05_09.jsonl"
LOG_DIR="/tmp/slashdot_edge_cr_5seed_2026_05_09"
mkdir -p "$LOG_DIR"
> "$RESULTS_FILE"

echo "[edge_cr] $(date +'%Y-%m-%d %H:%M:%S') start"

# Wait for any in-flight run_final_cell process to free the GPU.
echo "[edge_cr] waiting for GPU..."
while pgrep -af "python -m signedkan_wip.src.run_final_cell" \
        | grep -v "$$" | grep -q .; do
    sleep 60
done
echo "[edge_cr] $(date +%H:%M:%S) GPU free, starting"

run_cell() {
    local seed="$1"
    local logf="$LOG_DIR/edge_cr_seed${seed}.log"
    local t0=$(date +%s)
    echo "[edge_cr] $(date +%H:%M:%S) START seed=$seed"
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
d['run_label'] = 'edge_cr'
d['elapsed_s'] = $elapsed
print(json.dumps(d))" >> "$RESULTS_FILE"
        local auc
        auc=$(echo "$result" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        echo "[edge_cr] $(date +%H:%M:%S) OK    seed=$seed AUC=$auc (${elapsed}s)"
    else
        echo "[edge_cr] $(date +%H:%M:%S) FAIL  seed=$seed (see $logf)"
    fi
}

for seed in 0 1 2 3 4; do
    run_cell "$seed"
done

echo "[edge_cr] $(date +'%H:%M:%S') DONE"

python - <<'PY'
import json, statistics, pathlib
new = pathlib.Path("signedkan_wip/experiments/results/slashdot_edge_cr_5seed_2026_05_09.jsonl")
old = pathlib.Path("signedkan_wip/experiments/results/slashdot_c5_5seed_2026_05_08.jsonl")
old_aux = pathlib.Path("signedkan_wip/experiments/results/slashdot_aux_entropy_2026_05_09.jsonl")

def load(p):
    if not p.exists(): return []
    return [json.loads(l)["auc"] for l in p.read_text().splitlines() if l.strip()]

new_aucs = load(new)
old_aucs = load(old)
# Filter aux to aux_a1 only.
old_aux_aucs_a1 = []
old_aux_aucs_a1_a2 = []
for line in old_aux.read_text().splitlines() if old_aux.exists() else []:
    if not line.strip(): continue
    r = json.loads(line)
    if r["run_label"] == "aux_a1":
        old_aux_aucs_a1.append(r["auc"])
    elif r["run_label"] == "aux_a1_a2":
        old_aux_aucs_a1_a2.append(r["auc"])

print()
def report(label, aucs):
    if not aucs: return
    m = statistics.mean(aucs); s = (statistics.stdev(aucs)
                                     if len(aucs) > 1 else 0.0)
    print(f"{label:<28} n={len(aucs)}  mean={m:.4f}  std={s:.4f}  "
          f"per-seed: {','.join(f'{a:.4f}' for a in aucs)}")
report("c5full (2026-05-08)",   old_aucs)
report("aux_a1 (2026-05-09)",   old_aux_aucs_a1)
report("aux_a1_a2",             old_aux_aucs_a1_a2)
report("edge_cr (today)",       new_aucs)

# Paired Δ vs c5full baseline.
if len(new_aucs) == len(old_aucs):
    deltas = [a - b for a, b in zip(new_aucs, old_aucs)]
    md = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    sigma = md * (len(deltas) ** 0.5) / max(sd, 1e-9)
    print(f"  edge_cr vs c5full paired Δ = {md:+.4f} ± {sd:.4f}  "
          f"σ-units = {sigma:+.2f}")
PY
