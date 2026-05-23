#!/usr/bin/env bash
# Phase 22 — 5-seed paired A/B of SideMixedAritySignedKAN N=1 vs N=2 on
# Slashdot at the edge_cr highway SOTA config (kernel ON).
#
# Hypothesis: at Slashdot's σ=0.0029 baseline (7× tighter than Phase 21's
# Bitcoin Alpha σ=0.0005), the variance-tightening lever from Phase 19
# (Side σ ≈ 0.013 uniformly on c3-only) may still have slack to act on.
# Phase 21 falsified this at the Bitcoin Alpha 0.997 / σ=0.0005 ceiling.
#
# N=4 is GPU-bound on the 7.6 GiB card (sequential-branch forward state
# is too big with quaternion attention on, which auto-disables
# cycle_batch_size). N=2 is the largest fit; grad-checkpoint refactor
# would unlock N=4 but is deferred until Phase 22 shows directional
# signal at N=2.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/phase22_slashdot_5seed_2026_05_20.jsonl"
LOG_DIR="/tmp/phase22_slashdot_5seed_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

echo "[p22] $(date -Is) START stamp=$STAMP log_dir=$LOG_DIR" \
  | tee -a "$LOG_DIR/orchestrator.log"

export HYMEKO_CYCLE_CACHE=1

run_cell() {
  local n_branches="$1"; shift
  local seed="$1"; shift
  local label="N${n_branches}"
  local logf="$LOG_DIR/${label}_seed${seed}.log"
  local t0; t0=$(date +%s)
  echo "[p22] $(date -Is) START $label seed=$seed" \
    | tee -a "$LOG_DIR/orchestrator.log"
  systemd-run --user --scope -p MemoryMax=14G \
    env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_TRITON_KERNEL=1 HSIKAN_TRITON_BACKWARD=1 \
        HSIKAN_MAX_K3=200000 HSIKAN_MAX_K2=200000 \
        HSIKAN_CYCLE_BATCH=1000 \
        HYMEKO_CYCLE_CACHE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m signedkan_wip.experiments.runs.run_final_cell \
      --dataset slashdot --hidden 4 --n-epochs 80 \
      --max-k4 200000 --seed "$seed" \
      --n-branches "$n_branches" --side-fusion mean \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  local result
  result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
  if [ -n "$result" ] && [ $rc -eq 0 ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['n_branches'] = $n_branches
d['phase22_label'] = '$label'
d['elapsed_s'] = $elapsed
print(json.dumps(d))
" >> "$RESULTS_FILE"
    local auc
    auc=$(python -c "import json; print(f'{json.loads('''$result''')[\"auc\"]:.4f}')")
    echo "[p22] $(date -Is) DONE  $label seed=$seed AUC=$auc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[p22] $(date -Is) FAIL  $label seed=$seed rc=$rc" \
      | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

# 5 seeds x 2 configs (N=1, 2). N=4 is GPU-bound on 7.6 GiB.
for SEED in 0 1 2 3 4; do
  for N in 1 2; do
    run_cell "$N" "$SEED"
  done
done

echo "[p22] $(date -Is) DONE all 10 runs; jsonl=$RESULTS_FILE" \
  | tee -a "$LOG_DIR/orchestrator.log"

# Paired aggregate.
python - <<PY
import json, statistics, pathlib, math
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
by_n = {}
for r in rows:
    by_n.setdefault(r["n_branches"], []).append(r)
print(f"\n=== Phase 22 Slashdot 5-seed summary ===")
print(f"{'N':>4}  {'mean AUC':>12}  {'sigma':>8}  {'wall (s)':>10}  {'n_params':>10}")
print("-" * 60)
for n in sorted(by_n):
    cell = by_n[n]
    aucs = [r["auc"] for r in cell]
    walls = [r["elapsed_s"] for r in cell]
    nparams = cell[0].get("n_params", -1)
    mu = statistics.mean(aucs)
    sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    print(f"  N={n:<3} {mu:>10.4f}  {sd:>8.4f}  {statistics.mean(walls):>10.1f}  {nparams:>10,}")
print()
base = by_n.get(1, [])
base_by_seed = {r["seed"]: r["auc"] for r in base}
for n in sorted(by_n):
    if n == 1: continue
    deltas = []
    for r in by_n[n]:
        if r["seed"] in base_by_seed:
            deltas.append(r["auc"] - base_by_seed[r["seed"]])
    if not deltas: continue
    mu_d = statistics.mean(deltas)
    sd_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    se_d = sd_d / math.sqrt(len(deltas)) if sd_d > 0 else 0.0
    z = mu_d / se_d if se_d > 0 else float("inf")
    wins = sum(1 for d in deltas if d > 0)
    print(f"  N={n} vs N=1 paired Δ = {mu_d:+.4f} ± {sd_d:.4f}  σ_d={z:+.2f}  wins={wins}/{len(deltas)}")
PY
