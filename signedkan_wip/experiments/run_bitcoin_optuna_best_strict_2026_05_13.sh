#!/usr/bin/env bash
# Strict-protocol companion to bitcoin_optuna_best_5seed_2026_05_13:
# does optuna_best survive HSIKAN_STRICT_PROTOCOL=1?
#
# Predicted outcome (from code-reading of run_final_cell.py:244-345):
# both Alpha and OTC collapse to exactly 0.5000 across all seeds, like
# joint_mix_strict already did in joint_mix_5seed_2026_05_08.jsonl.
# The smoke verifies the code-reading; the 5-seed completion only fires
# if seed=0 clears 0.55 (well above random).
#
# CLAUDE.md §3 production-scale smoke before multi-seed run.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

OUT="signedkan_wip/experiments/results/bitcoin_optuna_best_strict_2026_05_13.jsonl"
LOG_DIR="/tmp/bitcoin_optuna_best_strict_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
: > "$OUT"
MASTER="$LOG_DIR/orchestrator.log"
echo "[strict] $(date -Is) START" | tee -a "$MASTER"

# Wait for any GPU-occupying signedkan_wip process.
echo "[strict] waiting for GPU..." | tee -a "$MASTER"
SELF_PID="$$"
while pgrep -af 'signedkan_wip\.src\.run_optuna_search|signedkan_wip\.src\.run_final_cell|signedkan_wip\.src\.vision\.train' \
      | grep -vE "^${SELF_PID} |^$0$" | grep -vF "$0" | grep -q .; do
  sleep 60
done
echo "[strict] GPU free $(date -Is)" | tee -a "$MASTER"

export HYMEKO_CYCLE_CACHE=1
export HSIKAN_STRICT_PROTOCOL=1

run_cell() {
  local label="$1"; shift
  local seed="$1"; shift
  local dataset="$1"; shift
  local hidden="$1"; shift
  local cap="$1"; shift
  local logf="$LOG_DIR/${label}_seed${seed}.log"
  local t0; t0=$(date +%s)
  echo "[strict] $(date -Is) START $label seed=$seed" | tee -a "$MASTER"
  env "$@" \
    HYMEKO_CYCLE_CACHE=1 HSIKAN_STRICT_PROTOCOL=1 HSIKAN_CYCLE_BATCH=2000 \
    python -m signedkan_wip.experiments.runs.run_final_cell \
      --dataset "$dataset" --hidden "$hidden" --n-epochs 80 \
      --max-k4 "$cap" --seed "$seed" \
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
d['protocol'] = 'strict'
print(json.dumps(d))
" >> "$OUT"
    local auc
    auc=$(echo "$result" | python -c 'import sys,json; print(f"{json.loads(sys.stdin.read())[\"auc\"]:.4f}")')
    echo "[strict] $(date -Is) DONE  $label seed=$seed AUC=$auc (${elapsed}s)" | tee -a "$MASTER"
    echo "$auc" > "$LOG_DIR/${label}_seed${seed}.auc"
  else
    echo "[strict] $(date -Is) FAIL  $label seed=$seed rc=$rc (${elapsed}s)" | tee -a "$MASTER"
  fi
}

# Phase 1 smoke: seed=0 for both datasets.
run_cell "optuna_best_alpha_strict" 0 "bitcoin_alpha" 8  100000 \
  "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
  "HSIKAN_MAX_K3=100000" "HSIKAN_MAX_K2=100000" \
  "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301"

run_cell "optuna_best_otc_strict" 0 "bitcoin_otc" 4 50000 \
  "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
  "HSIKAN_MAX_K3=50000" "HSIKAN_MAX_K2=50000" \
  "HSIKAN_ATTENTION_M_E=quaternion" \
  "HSIKAN_ATTENTION_HIGHWAY=1" \
  "HSIKAN_ATTENTION_HIGHWAY_MAX=0.13682674286852775" \
  "HSIKAN_ALPHA_ENTROPY_LAMBDA=1.4777880758638605e-05" \
  "HSIKAN_ATTN_ENTROPY_LAMBDA=0.0012729880784274699"

# Gate Phase 2 on min(seed=0 alpha, seed=0 otc) >= 0.55
ALPHA_S0=$(cat "$LOG_DIR/optuna_best_alpha_strict_seed0.auc" 2>/dev/null || echo "0.0")
OTC_S0=$(cat "$LOG_DIR/optuna_best_otc_strict_seed0.auc" 2>/dev/null || echo "0.0")
echo "[strict] $(date -Is) GATE seed0: alpha=$ALPHA_S0 otc=$OTC_S0" | tee -a "$MASTER"

# bc-free numeric compare via python
GATE=$(python -c "print('go' if min(float('$ALPHA_S0'), float('$OTC_S0')) >= 0.55 else 'halt')")
if [ "$GATE" = "halt" ]; then
  echo "[strict] $(date -Is) HALT — seed=0 below 0.55 threshold; consistent with predicted protocol-collapse. Not running seeds 1-4." | tee -a "$MASTER"
else
  echo "[strict] $(date -Is) PROCEED — seed=0 cleared 0.55; running seeds 1-4." | tee -a "$MASTER"
  for SEED in 1 2 3 4; do
    run_cell "optuna_best_alpha_strict" "$SEED" "bitcoin_alpha" 8 100000 \
      "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
      "HSIKAN_MAX_K3=100000" "HSIKAN_MAX_K2=100000" \
      "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301"
    run_cell "optuna_best_otc_strict" "$SEED" "bitcoin_otc" 4 50000 \
      "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
      "HSIKAN_MAX_K3=50000" "HSIKAN_MAX_K2=50000" \
      "HSIKAN_ATTENTION_M_E=quaternion" \
      "HSIKAN_ATTENTION_HIGHWAY=1" \
      "HSIKAN_ATTENTION_HIGHWAY_MAX=0.13682674286852775" \
      "HSIKAN_ALPHA_ENTROPY_LAMBDA=1.4777880758638605e-05" \
      "HSIKAN_ATTN_ENTROPY_LAMBDA=0.0012729880784274699"
  done
fi

# Aggregate
python - <<PY
import json, statistics, pathlib
rows = [json.loads(l) for l in pathlib.Path("$OUT").read_text().splitlines() if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["run_label"], []).append(r["auc"])
for label, aucs in sorted(by.items()):
    if len(aucs) >= 2:
        print(f"  {label:<28s}  n={len(aucs):<2d}  mean={statistics.mean(aucs):.4f}  pstdev={statistics.pstdev(aucs):.4f}")
    else:
        print(f"  {label:<28s}  n=1  AUC={aucs[0]:.4f}  (smoke only — gate did not clear)")
PY

echo "[strict] $(date -Is) END" | tee -a "$MASTER"
