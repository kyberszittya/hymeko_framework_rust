#!/usr/bin/env bash
# Phase 21 — 5-seed paired A/B of SideMixedAritySignedKAN at N∈{1, 4, 8}
# on Bitcoin Alpha (Optuna best config, AUC SOTA 0.9959 ± .0011).
#
# Tests whether N parallel mixed-arity branches with mean-fusion beats
# the single-branch SOTA at iso-architecture (same c2,c5,w2,w3,w4 mix,
# same hidden=8, same epochs=80).
#
# Headlines we expect:
#   - N=1 reproduces the baseline single-branch result (sanity)
#   - N=4 / N=8 either lifts AUC by some thousandths AND/OR tightens
#     σ (Phase 17/19 result: Side σ ≈ 0.013 uniformly)
#   - The gap is ~ 0.014 on c3-only @ low-AUC regime; at 0.996 we
#     expect smaller absolute lift but tighter σ.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/phase21_side_mixed_5seed_2026_05_20.jsonl"
LOG_DIR="/tmp/phase21_side_mixed_5seed_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

echo "[p21] $(date -Is) START stamp=$STAMP log_dir=$LOG_DIR" \
  | tee -a "$LOG_DIR/orchestrator.log"

export HYMEKO_CYCLE_CACHE=1
export HSIKAN_CYCLE_BATCH=2000

run_cell() {
  local n_branches="$1"; shift
  local seed="$1"; shift
  local label="N${n_branches}"
  local logf="$LOG_DIR/${label}_seed${seed}.log"
  local t0; t0=$(date +%s)
  echo "[p21] $(date -Is) START $label seed=$seed" \
    | tee -a "$LOG_DIR/orchestrator.log"
  systemd-run --user --scope -p MemoryMax=14G \
    env HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4 \
        HSIKAN_MAX_K3=100000 HSIKAN_MAX_K2=100000 \
        HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301 \
        HYMEKO_CYCLE_CACHE=1 HSIKAN_CYCLE_BATCH=2000 \
    python -m signedkan_wip.experiments.runs.run_final_cell \
      --dataset bitcoin_alpha --hidden 8 --n-epochs 80 \
      --max-k4 100000 --seed "$seed" \
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
d['phase21_label'] = '$label'
d['elapsed_s'] = $elapsed
print(json.dumps(d))
" >> "$RESULTS_FILE"
    local auc
    auc=$(echo "$result" | python -c 'import sys,json;print(f"{json.loads(sys.stdin.read())[\"auc\"]:.4f}")')
    echo "[p21] $(date -Is) DONE  $label seed=$seed AUC=$auc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[p21] $(date -Is) FAIL  $label seed=$seed rc=$rc" \
      | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

# 5 seeds x 2 configs (N=1, 4). N=8 dropped from the first A/B because
# the seed-0 smoke showed N=4 ≈ N=1 at the 0.997 ceiling; the wall cost
# of N=8 (5 × ~25 min = ~2 h) is not justified until N=4 demonstrates
# a lift or a meaningful variance-tightening signal first.
for SEED in 0 1 2 3 4; do
  for N in 1 4; do
    run_cell "$N" "$SEED"
  done
done

echo "[p21] $(date -Is) DONE all 10 runs; jsonl=$RESULTS_FILE" \
  | tee -a "$LOG_DIR/orchestrator.log"

# Quick aggregate (paired vs N=1 baseline).
python - <<PY
import json, statistics, pathlib, math
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
by_n = {}
for r in rows:
    by_n.setdefault(r["n_branches"], []).append(r)
print(f"\n=== Phase 21 5-seed summary (Bitcoin Alpha) ===")
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
# Paired vs N=1.
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
