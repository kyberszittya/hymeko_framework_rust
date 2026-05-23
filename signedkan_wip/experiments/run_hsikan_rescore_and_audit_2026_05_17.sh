#!/usr/bin/env bash
# Nature Comm supplementary evidence: Task A + Task B.
#
# Task A — HSiKAN-Optuna per-class P/R rescore (10-seed, --emit-full-metrics).
#   Replicates the exact Optuna best configs from
#   run_bitcoin_optuna_best_5seed_2026_05_13.sh on Bitcoin Alpha + OTC, but
#   adds the --emit-full-metrics flag patched into run_final_cell.py so each
#   run emits test_accuracy / test_precision_pos / test_recall_pos /
#   test_precision_neg / test_recall_neg / test_n_pos / test_n_neg alongside
#   AUC + F1_macro. Lets the Nature paper quote per-class P/R for the
#   imbalanced-negative regime (Bitcoin is ~93% positive).
#
# Task B — Label-shuffle audit (1 seed per dataset, Gömb-strict).
#   Reuses run_gomb_smoke.py with --shuffle-train-signs on Bitcoin OTC,
#   Slashdot, and Epinions, using the Optuna-tuned configs from
#   run_gomb_strict_benchmark_2026_05_14.sh. Confirms AUROC collapses to
#   chance under randomised TRAIN signs (already verified on Alpha at
#   gomb_strict_benchmark_20260514T005336Z/step0_shuffle_alpha_seed0.log
#   val=0.5692, test=0.5402; this run extends the audit to the other three).
#
# Total budget: ~4h GPU (A ~1h, B ~3h). Run sequentially.
#
# Launch:
#   bash signedkan_wip/experiments/run_hsikan_rescore_and_audit_2026_05_17.sh \
#     > /tmp/hsikan_rescore_audit_2026_05_17.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

# Fragmentation control (matches the strict-benchmark precedent).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/hsikan_rescore_audit_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
RESCORE_JSONL="${OUT_DIR}/task_a_hsikan_rescore.jsonl"
AUDIT_JSONL="${OUT_DIR}/task_b_shuffle_audit.jsonl"
: > "$RESCORE_JSONL"
: > "$AUDIT_JSONL"

echo "=== HSiKAN rescore + Gömb shuffle audit ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight HSiKAN / Optuna / Gömb training.
echo "[orch] waiting for GPU (any signedkan_wip training)..." | tee -a "$LOG"
while pgrep -af 'signedkan_wip\.src\.(run_optuna_search|run_final_cell|run_gomb_smoke)' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

export HYMEKO_CYCLE_CACHE=1

# ─── Task A — HSiKAN-Optuna rescore with full metrics ─────────────────

run_rescore() {
  local label="$1"; shift
  local seed="$1"; shift
  local dataset="$1"; shift
  local hidden="$1"; shift
  local cap="$1"; shift
  local logf="${OUT_DIR}/A_${label}_seed${seed}.log"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] A START $label seed=$seed h=$hidden cap=$cap" | tee -a "$LOG"
  env "$@" \
    HYMEKO_CYCLE_CACHE=1 \
    HSIKAN_CYCLE_BATCH=2000 \
    python -m signedkan_wip.experiments.runs.run_final_cell \
      --dataset "$dataset" --hidden "$hidden" --n-epochs 80 \
      --max-k4 "$cap" --seed "$seed" --emit-full-metrics \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  local result
  result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
  if [ -n "$result" ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = '$label'
d['elapsed_s'] = $elapsed
d['stamp'] = '$STAMP'
print(json.dumps(d))
" >> "$RESCORE_JSONL"
    local auc acc
    auc=$(echo "$result" | python -c 'import sys,json;print(f"{json.loads(sys.stdin.read())[\"auc\"]:.4f}")')
    acc=$(echo "$result" | python -c 'import sys,json;d=json.loads(sys.stdin.read());print(f"{d.get(\"test_accuracy\",float(\"nan\")):.4f}")')
    echo "[$(date -Is)] A DONE  $label seed=$seed AUC=$auc acc=$acc elapsed=${elapsed}s" | tee -a "$LOG"
  else
    echo "[$(date -Is)] A FAIL  $label seed=$seed rc=$rc (no JSON in $logf)" | tee -a "$LOG"
  fi
}

echo "" | tee -a "$LOG"
echo "## Task A — HSiKAN-Optuna 10-seed rescore (per-class P/R)" | tee -a "$LOG"

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  # Alpha trial 23: c2,c5,w2,w3,w4 h=8 cap=100000 lam_a=0.0966
  run_rescore "alpha" "$SEED" "bitcoin_alpha" 8 100000 \
    "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
    "HSIKAN_MAX_K3=100000" "HSIKAN_MAX_K2=100000" \
    "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301"

  # OTC trial 28: c2,c5,w2,w3,w4 h=4 cap=50000 attn=quat hw=0.137 lam_a=1.48e-5 lam_attn=1.27e-3
  run_rescore "otc" "$SEED" "bitcoin_otc" 4 50000 \
    "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
    "HSIKAN_MAX_K3=50000" "HSIKAN_MAX_K2=50000" \
    "HSIKAN_ATTENTION_M_E=quaternion" \
    "HSIKAN_ATTENTION_HIGHWAY=1" \
    "HSIKAN_ATTENTION_HIGHWAY_MAX=0.13682674286852775" \
    "HSIKAN_ALPHA_ENTROPY_LAMBDA=1.4777880758638605e-05" \
    "HSIKAN_ATTN_ENTROPY_LAMBDA=0.0012729880784274699"
done

# Quick aggregate for Task A
python - <<PY | tee -a "$LOG"
import json, statistics, pathlib
rows = [json.loads(l) for l in pathlib.Path("$RESCORE_JSONL").read_text().splitlines() if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["run_label"], []).append(r)
print("[Task A aggregate]")
for label, items in sorted(by.items()):
    keys = ["auc", "test_accuracy", "test_precision_pos", "test_recall_pos",
            "test_precision_neg", "test_recall_neg", "test_f1_macro"]
    line = f"  {label}: n={len(items)}"
    for k in keys:
        vals = [it.get(k) for it in items if it.get(k) is not None]
        if vals:
            line += f"  {k}={statistics.mean(vals):.4f}±{statistics.pstdev(vals):.4f}"
    print(line)
PY

# ─── Task B — Label-shuffle audit (Gömb-strict) ───────────────────────

run_audit() {
  local label="$1"; shift
  local dataset="$1"; shift
  local epochs="$1"; shift
  local extra_args="$@"
  local seed=0
  local logf="${OUT_DIR}/B_${label}_seed${seed}_shuffle.log"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] B START $label dataset=$dataset epochs=$epochs (shuffled TRAIN signs)" | tee -a "$LOG"
  python -m signedkan_wip.experiments.runs.run_gomb_smoke \
    --dataset "$dataset" --seed "$seed" \
    --n-epochs "$epochs" \
    --edge-split 80_10_10 --joint-mix \
    --shuffle-train-signs \
    --device cuda \
    $extra_args \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  local result
  result=$(grep -E '^\{"dataset"' "$logf" | tail -1)
  if [ -n "$result" ]; then
    echo "$result" | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['run_label'] = '$label'
d['shuffled'] = True
d['elapsed_s'] = $elapsed
d['stamp'] = '$STAMP'
print(json.dumps(d))
" >> "$AUDIT_JSONL"
    echo "[$(date -Is)] B DONE  $label rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"
    echo "  result: $result" | tee -a "$LOG"
  else
    echo "[$(date -Is)] B FAIL  $label rc=$rc (no JSON in $logf)" | tee -a "$LOG"
    grep -E "val_auroc|test_auroc|Error|OutOfMemoryError" "$logf" | tail -3 | tee -a "$LOG"
  fi
}

echo "" | tee -a "$LOG"
echo "## Task B — Gömb-strict label-shuffle audit (1 seed × 3 datasets)" | tee -a "$LOG"

# Bitcoin OTC — Optuna-tuned config
run_audit "otc" "bitcoin_otc" 80 \
  --M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 \
  --n-tiers 2 --topk 32 --lr 0.005

# Slashdot — slim SOTA config
run_audit "slashdot" "slashdot" 60 \
  --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
  --n-tiers 3 --topk 32 --lr 0.003

# Epinions — slim (same regime as Slashdot)
run_audit "epinions" "epinions" 60 \
  --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
  --n-tiers 3 --topk 32 --lr 0.003

# Quick aggregate for Task B
python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$AUDIT_JSONL").read_text().splitlines() if l.strip()]
print("[Task B aggregate — shuffled TRAIN signs; expect AUROC≈0.5]")
for r in rows:
    auc = r.get("test_auroc") or r.get("test_auc") or r.get("auc")
    print(f"  {r['run_label']}: test_auroc={auc}")
PY

echo "" | tee -a "$LOG"
echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG"
echo "  Task A jsonl: $RESCORE_JSONL" | tee -a "$LOG"
echo "  Task B jsonl: $AUDIT_JSONL" | tee -a "$LOG"
