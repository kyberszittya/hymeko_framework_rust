#!/usr/bin/env bash
# Outer HSIKAN → Clifford-FIR → Gömb cascade — overnight grid.
# 2026-05-20.
#
# 2 datasets × 3 outer-HSIKAN depths × 3 seeds = 18 cells. Plain
# Gömb baselines (depth=0 effectively = HymeKoGomb) are not
# rerun — the 2026-05-20-stacked-gomb-hsikan-backbone overnight
# already produced 3-seed plain-Gömb numbers at the strict-bench
# config we use here, so paired Δ vs that baseline is the
# tomorrow-morning comparison.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/outer_hsikan_gomb_overnight_2026_05_20.jsonl"
LOG_DIR="/tmp/outer_hsikan_gomb_overnight_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

echo "[ohg] $(date -Is) START stamp=$STAMP log_dir=$LOG_DIR" \
  | tee -a "$LOG_DIR/orchestrator.log"

run_cell() {
  local dataset="$1"; shift
  local depth="$1"; shift
  local seed="$1"; shift
  local n_epochs="$1"; shift
  local label="${dataset}_ohd${depth}_s${seed}"
  local logf="$LOG_DIR/${label}.log"
  local t0; t0=$(date +%s)
  echo "[ohg] $(date -Is) START $label epochs=$n_epochs" \
    | tee -a "$LOG_DIR/orchestrator.log"

  local extra=""
  if [ "$dataset" = "bitcoin_alpha" ] || [ "$dataset" = "bitcoin_otc" ]; then
    extra="--d-embed 32 --M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 --n-tiers 4 --topk 56 --lr 0.005 --pos-weight-auto"
  elif [ "$dataset" = "slashdot" ]; then
    extra="--d-embed 16 --M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 --n-tiers 2 --topk 32 --lr 0.005"
  fi

  systemd-run --user --scope -p MemoryMax=14G \
    env PATH="/home/kyberszittya/miniconda3/bin:$PATH" \
        PYTHONPATH="$REPO_ROOT" \
        HYMEKO_CYCLE_CACHE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
      --dataset "$dataset" --seed "$seed" \
      --n-epochs "$n_epochs" \
      --model outer_hsikan_gomb \
      --outer-hsikan-n-layers "$depth" \
      --outer-hsikan-inner-skip highway \
      --outer-hsikan-jk-mode last \
      $extra \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  local result
  result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
  if [ -n "$result" ] && [ $rc -eq 0 ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['outer_depth']  = $depth
d['oh_label']     = '$label'
d['elapsed_s']    = $elapsed
print(json.dumps(d))
" >> "$RESULTS_FILE"
    echo "[ohg] $(date -Is) DONE  $label rc=$rc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[ohg] $(date -Is) FAIL  $label rc=$rc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

# Bitcoin Alpha: 3 outer-depths × 3 seeds = 9 cells.
for DEPTH in 1 2 4; do
  for SEED in 0 1 2; do
    run_cell "bitcoin_alpha" "$DEPTH" "$SEED" 60
  done
done

# Slashdot: same grid.
for DEPTH in 1 2 4; do
  for SEED in 0 1 2; do
    run_cell "slashdot" "$DEPTH" "$SEED" 60
  done
done

echo "[ohg] $(date -Is) DONE all 18 runs; jsonl=$RESULTS_FILE" \
  | tee -a "$LOG_DIR/orchestrator.log"

python - <<PY
import json, statistics, pathlib, math
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]

# Plain-Gömb baseline from the earlier stacked-gomb-hsikan overnight
# (depth=1 = no middle stacking = plain HymeKoGomb at strict-bench).
base_path = pathlib.Path("signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl")
base_rows = []
if base_path.exists():
    for l in base_path.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("depth") == 1:
                base_rows.append(r)

print(f"\n=== Outer-HSIKAN Gömb overnight summary ===")
for ds in ["bitcoin_alpha", "slashdot"]:
    cells = [r for r in rows if r["dataset"] == ds]
    if not cells:
        print(f"\n--- {ds} ---  (no runs)")
        continue
    print(f"\n--- {ds} ---")
    print(f"{'outer_d':>8}  {'mean AUC':>10}  {'sigma':>8}  {'wall':>8}  n_seeds")
    by_depth = {}
    for r in cells:
        by_depth.setdefault(r["outer_depth"], []).append(r)
    for depth in sorted(by_depth):
        rs = by_depth[depth]
        aucs = [r.get("val_auc_best") or r.get("val_auroc") for r in rs]
        aucs = [a for a in aucs if a is not None]
        walls = [r["elapsed_s"] for r in rs]
        if not aucs: continue
        mu = statistics.mean(aucs)
        sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        print(f"  d={depth:<3} {mu:>10.4f}  {sd:>8.4f}  {statistics.mean(walls):>7.1f}s  {len(aucs)}")

    # Paired Δ vs plain Gömb (from the prior overnight).
    base_for_ds = [r for r in base_rows if r["dataset"] == ds]
    base_by_seed = {r["seed"]: (r.get("val_auc_best") or r.get("val_auroc"))
                     for r in base_for_ds}
    if base_by_seed:
        for d in sorted(by_depth):
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
            print(f"  d={d} vs PLAIN GÖMB paired Δ={mu_d:+.4f} ± {sd_d:.4f}  σ_d={z:+.2f}  wins={wins}/{len(deltas)}")
PY
