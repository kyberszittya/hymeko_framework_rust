#!/usr/bin/env bash
# Outer-HSIKAN residual highway-gated — 5-seed cross-dataset sweep.
#
# The BA d=4 cell delivered +0.0066 AUC, 5.68σ paired, 5/5 wins last
# week. The lever generalises directionally to Bitcoin OTC (+0.0045,
# 1.73σ at 5 seeds). This script firms up:
#
#   1. **Slashdot** outer-HSIKAN d=4 vs plain Gömb (5-seed paired). Was
#      only single-seed-explored before.
#   2. **Bitcoin OTC** outer-HSIKAN d=4 (full 5-seed; firms up the 1.73σ).
#   3. **Bitcoin Alpha** OUTER-HSIKAN d=8 (untested depth; tests whether
#      depth saturates beyond d=4).
#
# Queues behind the CV experiment via pgrep loop on train_voc_*.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/outer_hsikan_overnight_5seed_${STAMP}.jsonl"
LOG_DIR="/tmp/outer_hsikan_overnight_5seed_${STAMP}"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/orchestrator.log"
: > "$RESULTS_FILE"

echo "[ohn] $(date -Is) START stamp=$STAMP log_dir=$LOG_DIR" | tee -a "$ORCH_LOG"
echo "[ohn] $(date -Is) git=$(git rev-parse HEAD)" | tee -a "$ORCH_LOG"

# Queue behind the CV experiment (train_voc_stagec) so they don't
# share the 8 GiB GPU.
echo "[ohn] $(date -Is) waiting for CV experiment to finish..." | tee -a "$ORCH_LOG"
while pgrep -af 'train_voc_stagec' | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 60
done
echo "[ohn] $(date -Is) CV experiment finished, starting" | tee -a "$ORCH_LOG"

run_cell () {
  local dataset="$1"; shift
  local model="$1"; shift  # gomb | outer_hsikan_gomb
  local depth="$1"; shift  # outer HSIKAN depth (ignored for plain gomb)
  local seed="$1"; shift
  local label="${dataset}_${model}_d${depth}_s${seed}"
  local logf="$LOG_DIR/${label}.log"
  local t0; t0=$(date +%s)
  echo "[ohn] $(date -Is) START $label" | tee -a "$ORCH_LOG"

  # Per-dataset configs (matching the proven 2026-05-20 / 05-21 settings).
  local extra=""
  if [ "$dataset" = "bitcoin_alpha" ] || [ "$dataset" = "bitcoin_otc" ]; then
    extra="--d-embed 32 --M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 --n-tiers 4 --topk 56 --lr 0.005 --pos-weight-auto --n-epochs 60"
  elif [ "$dataset" = "slashdot" ]; then
    extra="--d-embed 16 --M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 --n-tiers 2 --topk 32 --lr 0.005 --n-epochs 50"
  fi

  local outer_flags=""
  if [ "$model" = "outer_hsikan_gomb" ]; then
    outer_flags="--outer-hsikan-n-layers $depth --outer-hsikan-inner-skip highway --outer-hsikan-jk-mode last"
  fi

  systemd-run --user --scope -p MemoryMax=14G --quiet \
    --unit="ohn_${STAMP}_${label}.scope" \
    env PATH="/home/kyberszittya/miniconda3/bin:$PATH" \
        PYTHONPATH="$REPO_ROOT" \
        HYMEKO_CYCLE_CACHE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
      --dataset "$dataset" --seed "$seed" \
      --model "$model" \
      $outer_flags \
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
d['outer_depth'] = $depth
d['ohn_label']   = '$label'
d['elapsed_s']   = $elapsed
d['composition'] = 'residual_highway' if '$model' == 'outer_hsikan_gomb' else 'plain'
print(json.dumps(d))
" >> "$RESULTS_FILE"
    local auc
    auc=$(echo "$result" | python -c "
import sys, json; print(f\"{json.loads(sys.stdin.read()).get('val_auroc', float('nan')):.4f}\")
" 2>&1)
    echo "[ohn] $(date -Is) DONE  $label rc=$rc elapsed=${elapsed}s val_auroc=$auc" | tee -a "$ORCH_LOG"
  else
    echo "[ohn] $(date -Is) FAIL  $label rc=$rc elapsed=${elapsed}s" | tee -a "$ORCH_LOG"
  fi
}

# Run order: smallest/cheapest first, so we have early signal even if
# the long ones are still going at wakeup.

# ─── Block 1: Bitcoin OTC, 5-seed paired (gomb + outer_hsikan d=4) ───
for SEED in 0 1 2 3 4; do
  run_cell "bitcoin_otc" "gomb" 0 "$SEED"
  run_cell "bitcoin_otc" "outer_hsikan_gomb" 4 "$SEED"
done

# ─── Block 2: Bitcoin Alpha d=8 depth probe (5-seed) ─────────────────
for SEED in 0 1 2 3 4; do
  run_cell "bitcoin_alpha" "outer_hsikan_gomb" 8 "$SEED"
done

# ─── Block 3: Slashdot, 5-seed paired (gomb + outer_hsikan d=4) ──────
for SEED in 0 1 2 3 4; do
  run_cell "slashdot" "gomb" 0 "$SEED"
  run_cell "slashdot" "outer_hsikan_gomb" 4 "$SEED"
done

# ─── Aggregate ───────────────────────────────────────────────────────
python - <<PY | tee -a "$ORCH_LOG"
import json, pathlib, statistics, collections
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
by_key = collections.defaultdict(list)
for r in rows:
    key = (r["dataset"], r.get("composition", "?"), r.get("outer_depth", 0))
    a = r.get("val_auroc")
    if isinstance(a, (int, float)):
        by_key[key].append(a)
print()
print(f"=== outer-HSIKAN-residual 5-seed cross-dataset summary ===")
print(f"{'dataset':16s} {'composition':18s} {'depth':>5s} {'n':>3s} {'mean':>8s} {'pstdev':>8s}")
for k in sorted(by_key.keys()):
    aucs = by_key[k]
    mean = statistics.mean(aucs)
    sd = statistics.pstdev(aucs) if len(aucs) >= 2 else 0.0
    print(f"{k[0]:16s} {k[1]:18s} {k[2]:>5d} {len(aucs):>3d} {mean:>8.4f} {sd:>8.4f}")

# Paired Δ where both arms have 5 seeds at matched dataset
print()
for ds in ("bitcoin_otc", "slashdot"):
    base = sorted(by_key.get((ds, "plain", 0), []))
    test = sorted(by_key.get((ds, "residual_highway", 4), []))
    if len(base) == 5 and len(test) == 5:
        diffs = [t-b for b, t in zip(base, test)]
        mean = statistics.mean(diffs)
        sd = statistics.pstdev(diffs)
        z = mean / (sd / (len(diffs) ** 0.5)) if sd > 0 else float('inf')
        wins = sum(1 for d in diffs if d > 0)
        print(f"[paired Δ] {ds:16s} d=4 - plain: mean={mean:+.4f} σ_d={z:+.2f} wins={wins}/5")
PY

echo "[ohn] $(date -Is) DONE  results=$RESULTS_FILE" | tee -a "$ORCH_LOG"
