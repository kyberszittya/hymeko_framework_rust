#!/usr/bin/env bash
# Cross-validate the outer-HSIKAN-residual lever on the wiki signed
# datasets: wikisigned + wiki_elec. Both are smaller than Slashdot;
# the Gömb-strict 5-seed numbers (from memory) are:
#   wikisigned   0.8944 ± 0.0019
#   wiki_elec    0.9114 ± 0.0013
#
# Two-grid layout:
#   1. Plain-Gömb 5-seed baseline (at Gömb-strict-bench config).
#   2. Outer-HSIKAN-residual d=4 (the BA d=4 winner) 5-seed.
# Paired Δ tells us whether the lever generalises beyond Bitcoin.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/wiki_outer_hsikan_grid_2026_05_21.jsonl"
LOG_DIR="/tmp/wiki_outer_hsikan_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

run_cell() {
  local dataset="$1"; shift
  local model="$1"; shift
  local seed="$1"; shift
  local extra="$1"; shift
  local label="${dataset}_${model}_s${seed}"
  local logf="$LOG_DIR/${label}.log"
  local t0; t0=$(date +%s)
  echo "[wog] $(date -Is) START $label" | tee -a "$LOG_DIR/orchestrator.log"
  systemd-run --user --scope -p MemoryMax=14G \
    env PATH="/home/kyberszittya/miniconda3/bin:$PATH" \
        PYTHONPATH="$REPO_ROOT" \
        HYMEKO_CYCLE_CACHE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
      --dataset "$dataset" --seed "$seed" --n-epochs 60 \
      --model "$model" \
      --d-embed 32 --M-outer 8 --d-outer 20 \
      --d-middle 24 --d-core 48 --n-tiers 4 \
      --topk 56 --lr 0.005 --pos-weight-auto \
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
d['variant'] = '$model'
d['cell_label'] = '$label'
d['elapsed_s'] = $elapsed
print(json.dumps(d))
" >> "$RESULTS_FILE"
    local auc; auc=$(echo "$result" | python -c 'import sys,json;print(f"{json.loads(sys.stdin.read())[\"val_auc_best\"]:.4f}")' 2>/dev/null)
    echo "[wog] DONE $label AUC=$auc ${elapsed}s" | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[wog] FAIL $label rc=$rc ${elapsed}s" | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

# 5 seeds × 2 datasets × 2 variants = 20 cells.
for DATASET in wikisigned wiki_elec; do
  for SEED in 0 1 2 3 4; do
    run_cell "$DATASET" "gomb" "$SEED" ""
    run_cell "$DATASET" "outer_hsikan_gomb" "$SEED" \
      "--outer-hsikan-n-layers 4 --outer-hsikan-inner-skip cr_highway --outer-hsikan-jk-mode last"
  done
done

echo "[wog] $(date -Is) DONE all 20 runs" | tee -a "$LOG_DIR/orchestrator.log"

# Aggregate with paired Δ.
python - <<PY
import json, statistics, pathlib, math
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
def auc(r): return r.get("val_auc_best") or r.get("val_auroc")
print("\n=== wiki signed cross-validation ===")
for ds in ("wikisigned", "wiki_elec"):
    plain = [r for r in rows if r["dataset"]==ds and r["variant"]=="gomb"]
    oh = [r for r in rows if r["dataset"]==ds and r["variant"]=="outer_hsikan_gomb"]
    if not plain or not oh: continue
    plain_by_seed = {r["seed"]: auc(r) for r in plain}
    plain_mu = statistics.mean(plain_by_seed.values())
    plain_sd = statistics.stdev(plain_by_seed.values()) if len(plain_by_seed)>1 else 0.0
    oh_aucs = [auc(r) for r in oh if auc(r) is not None]
    oh_mu = statistics.mean(oh_aucs)
    oh_sd = statistics.stdev(oh_aucs) if len(oh_aucs)>1 else 0.0
    deltas = [auc(r) - plain_by_seed[r["seed"]] for r in oh
              if auc(r) is not None and r["seed"] in plain_by_seed]
    mu_d = statistics.mean(deltas); sd_d = statistics.stdev(deltas) if len(deltas)>1 else 0.0
    se = sd_d / math.sqrt(len(deltas)) if sd_d > 0 else 0.0
    z = mu_d / se if se > 0 else float("inf")
    wins = sum(1 for x in deltas if x > 0)
    print(f"\n--- {ds} ---")
    print(f"  plain Gömb       : {plain_mu:.4f} ± {plain_sd:.4f}  n={len(plain_by_seed)}")
    print(f"  outer-HSIKAN d=4 : {oh_mu:.4f} ± {oh_sd:.4f}  n={len(oh_aucs)}")
    print(f"  paired Δ         : {mu_d:+.4f} ± {sd_d:.4f}  σ_d={z:+.2f}  wins={wins}/{len(deltas)}")
PY
