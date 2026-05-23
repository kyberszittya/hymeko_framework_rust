#!/usr/bin/env bash
# Slashdot-only retry of the stacked Gömb-HSIKAN overnight grid.
# Bitcoin Alpha cells already completed in the main 2026-05-20 run;
# this script only re-queues the 9 Slashdot cells (depth × seed) with
# the corrected GPU-budget-fitting hyperparameters.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/stacked_gomb_overnight_slashdot_2026_05_20.jsonl"
LOG_DIR="/tmp/stacked_gomb_overnight_slashdot_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

echo "[sgs] $(date -Is) START stamp=$STAMP log_dir=$LOG_DIR" \
  | tee -a "$LOG_DIR/orchestrator.log"

run_cell() {
  local depth="$1"; shift
  local seed="$1"; shift
  local n_epochs="$1"; shift
  local label="slashdot_d${depth}_s${seed}"
  local logf="$LOG_DIR/${label}.log"
  local t0; t0=$(date +%s)
  echo "[sgs] $(date -Is) START $label epochs=$n_epochs" \
    | tee -a "$LOG_DIR/orchestrator.log"

  systemd-run --user --scope -p MemoryMax=14G \
    env PATH="/home/kyberszittya/miniconda3/bin:$PATH" \
        PYTHONPATH="$REPO_ROOT" \
        HYMEKO_CYCLE_CACHE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
      --dataset slashdot --seed "$seed" \
      --n-epochs "$n_epochs" \
      --middle-n-layers "$depth" \
      --middle-inner-skip highway \
      --middle-jk-mode last \
      --d-embed 16 --M-outer 12 --d-outer 8 \
      --d-middle 16 --d-core 32 \
      --n-tiers 2 --topk 32 --lr 0.005 \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  local result
  result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
  if [ -n "$result" ] && [ $rc -eq 0 ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['depth']      = $depth
d['stacked_label'] = '$label'
d['elapsed_s'] = $elapsed
print(json.dumps(d))
" >> "$RESULTS_FILE"
    echo "[sgs] $(date -Is) DONE  $label rc=$rc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[sgs] $(date -Is) FAIL  $label rc=$rc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

for DEPTH in 1 2 4; do
  for SEED in 0 1 2; do
    run_cell "$DEPTH" "$SEED" 60
  done
done

echo "[sgs] $(date -Is) DONE all 9 runs; jsonl=$RESULTS_FILE" \
  | tee -a "$LOG_DIR/orchestrator.log"

python - <<PY
import json, statistics, pathlib, math
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
print(f"\n=== Slashdot stacked Gömb-HSIKAN summary ===")
print(f"{'depth':>6}  {'mean AUC':>10}  {'sigma':>8}  {'wall':>8}  n_seeds")
by_depth = {}
for r in rows:
    by_depth.setdefault(r["depth"], []).append(r)
for depth in sorted(by_depth):
    rs = by_depth[depth]
    aucs = [r.get("val_auc_best") or r.get("val_auroc") for r in rs]
    aucs = [a for a in aucs if a is not None]
    walls = [r["elapsed_s"] for r in rs]
    if not aucs: continue
    mu = statistics.mean(aucs)
    sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    print(f"  d={depth:<3} {mu:>10.4f}  {sd:>8.4f}  {statistics.mean(walls):>7.1f}s  {len(aucs)}")
base = by_depth.get(1, [])
base_by_seed = {r["seed"]: (r.get("val_auc_best") or r.get("val_auroc")) for r in base}
for d in [2, 4]:
    if d not in by_depth: continue
    deltas = []
    for r in by_depth[d]:
        a = r.get("val_auc_best") or r.get("val_auroc")
        base_a = base_by_seed.get(r["seed"])
        if a is None or base_a is None: continue
        deltas.append(a - base_a)
    if not deltas: continue
    mu_d = statistics.mean(deltas)
    sd_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    se = sd_d / math.sqrt(len(deltas)) if sd_d > 0 else 0.0
    z = mu_d / se if se > 0 else float("inf")
    wins = sum(1 for x in deltas if x > 0)
    print(f"  d={d} vs d=1 paired Δ={mu_d:+.4f} ± {sd_d:.4f}  σ_d={z:+.2f}  wins={wins}/{len(deltas)}")
PY
