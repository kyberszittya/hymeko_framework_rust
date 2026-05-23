#!/usr/bin/env bash
# Stacked Gömb-HSIKAN backbone — overnight 2-dataset × 3-depth × 3-seed
# A/B grid. 2026-05-20.
#
# The middle shell of Gömb's three-shell cascade becomes a stacked
# MultiLayerSignedKAN (the Bitcoin-Optuna HSIKAN-SOTA stack). Outer
# (Clifford-FIR) and inner (CPML) shells stay unchanged. Depth axis:
# {1, 2, 4}. Baseline (depth=1) dispatches to the existing
# MiddleHSiKAN, exactly reproducing the current Gömb config.
#
# Per [[project-gomb-strict-4dataset-2026-05-14]] the strict-bench
# reference numbers are:
#   bitcoin_alpha   0.8972 ± _____
#   slashdot        0.9017 ± _____
# A depth-2 or depth-4 lift would be a directional positive for
# the stacked-HSIKAN hypothesis inside Gömb's cortical hierarchy.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl"
LOG_DIR="/tmp/stacked_gomb_overnight_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

echo "[sgo] $(date -Is) START stamp=$STAMP log_dir=$LOG_DIR" \
  | tee -a "$LOG_DIR/orchestrator.log"

run_cell() {
  local dataset="$1"; shift
  local depth="$1"; shift
  local seed="$1"; shift
  local n_epochs="$1"; shift
  local label="${dataset}_d${depth}_s${seed}"
  local logf="$LOG_DIR/${label}.log"
  local t0; t0=$(date +%s)
  echo "[sgo] $(date -Is) START $label epochs=$n_epochs" \
    | tee -a "$LOG_DIR/orchestrator.log"

  # Dataset-specific config (matches Gömb-strict-bench 2026-05-14).
  local extra=""
  if [ "$dataset" = "bitcoin_alpha" ] || [ "$dataset" = "bitcoin_otc" ]; then
    extra="--d-embed 32 --M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 --n-tiers 4 --topk 56 --lr 0.005 --pos-weight-auto"
  elif [ "$dataset" = "slashdot" ]; then
    # Slashdot-strict-bench dims from 2026-05-14 — fits the 7.6 GiB
    # GPU. Larger dims OOM at the CPML _edge_logits cat (1+ GiB
    # allocation on n_edges*2 query pairs).
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
      --middle-n-layers "$depth" \
      --middle-inner-skip highway \
      --middle-jk-mode last \
      $extra \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  # The script emits a single JSON line at the bottom with the run
  # summary; harvest it.
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
    local auc
    auc=$(echo "$result" | python -c 'import sys,json;d=json.loads(sys.stdin.read());print(f"{d.get(\"val_auc_best\", d.get(\"val_auroc\", float(\"nan\"))):.4f}")' 2>/dev/null)
    echo "[sgo] $(date -Is) DONE  $label rc=$rc AUC=$auc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[sgo] $(date -Is) FAIL  $label rc=$rc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

# Bitcoin Alpha — fast, 60 epochs, all 3 seeds × all 3 depths.
for DEPTH in 1 2 4; do
  for SEED in 0 1 2; do
    run_cell "bitcoin_alpha" "$DEPTH" "$SEED" 60
  done
done

# Slashdot — slower, 60 epochs, all 3 seeds × all 3 depths.
for DEPTH in 1 2 4; do
  for SEED in 0 1 2; do
    run_cell "slashdot" "$DEPTH" "$SEED" 60
  done
done

echo "[sgo] $(date -Is) DONE all 18 runs; jsonl=$RESULTS_FILE" \
  | tee -a "$LOG_DIR/orchestrator.log"

# Quick aggregate.
python - <<PY
import json, statistics, pathlib, math
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
print(f"\n=== Stacked Gömb-HSIKAN overnight summary ===")
for ds in ["bitcoin_alpha", "slashdot"]:
    cells = [r for r in rows if r["dataset"] == ds]
    if not cells:
        print(f"\n--- {ds} ---  (no runs)")
        continue
    print(f"\n--- {ds} ---")
    print(f"{'depth':>6}  {'mean AUC':>10}  {'sigma':>8}  {'wall':>8}  n_seeds")
    by_depth = {}
    for r in cells:
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
    # Paired delta vs depth=1.
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
