#!/usr/bin/env bash
# 5-seed paired validation of the 2026-05-13 Optuna best configs on
# Bitcoin Alpha + Bitcoin OTC vs the existing joint_mix 5-seed baseline.
#
# Tests whether the 0.9972 / 0.9957 single-trial maxima from
#   - bitcoin_alpha_20260513T010510Z  (trial 23, AUC=0.99722)
#   - bitcoin_otc_20260513T010159Z    (trial 28, AUC=0.99566)
# survive a 5-seed paired sweep at iso-protocol-non-iso-param vs
# joint_mix_5seed_2026_05_08.jsonl (h=16, mix=c3,c4,w2,w3).
#
# IMPORTANT: hidden differs (Alpha-best h=8, OTC-best h=4) so this is a
# strictly lower-param comparison: a win is also a leanness win.
#
# Queues behind any in-flight signedkan_wip.src.run_optuna_search via the
# same pgrep loop as run_overnight_joint_mix_2026_05_08.sh.
#
# Cache is enabled so multi-seed runs amortize cycle enumeration.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# Force the miniconda3 env (per memory reference_python_envs_for_optuna).
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_FILE="signedkan_wip/experiments/results/bitcoin_optuna_best_5seed_2026_05_13.jsonl"
LOG_DIR="/tmp/bitcoin_optuna_best_5seed_${STAMP}"
mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

echo "[obb] $(date -Is) START stamp=$STAMP" | tee -a "$LOG_DIR/orchestrator.log"

# Wait for Optuna search (and the per-trial run_final_cell) to release GPU.
echo "[obb] waiting for any run_optuna_search / run_final_cell..." | tee -a "$LOG_DIR/orchestrator.log"
while pgrep -af 'signedkan_wip\.src\.run_optuna_search|signedkan_wip\.src\.run_final_cell' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[obb] GPU free, starting paired sweep $(date -Is)" | tee -a "$LOG_DIR/orchestrator.log"

# Cycle cache on (cache is keyed off graph + topk fingerprint + arity + cap;
# enum_seed is decoupled from model seed so all 5 seeds hit the same key).
export HYMEKO_CYCLE_CACHE=1

run_cell() {
  local label="$1"; shift
  local seed="$1"; shift
  local dataset="$1"; shift
  local hidden="$1"; shift
  local cap="$1"; shift
  # Remaining args: K=V env tokens.
  local logf="$LOG_DIR/${label}_seed${seed}.log"
  local t0; t0=$(date +%s)
  echo "[obb] $(date -Is) START $label seed=$seed dataset=$dataset h=$hidden cap=$cap" \
    | tee -a "$LOG_DIR/orchestrator.log"
  env "$@" \
    HYMEKO_CYCLE_CACHE=1 \
    HSIKAN_CYCLE_BATCH=2000 \
    python -m signedkan_wip.src.run_final_cell \
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
d['optuna_stamp'] = '20260513T010510Z'
print(json.dumps(d))
" >> "$RESULTS_FILE"
    local auc
    auc=$(echo "$result" | python -c 'import sys,json;print(f"{json.loads(sys.stdin.read())[\"auc\"]:.4f}")')
    echo "[obb] $(date -Is) DONE  $label seed=$seed AUC=$auc elapsed=${elapsed}s" \
      | tee -a "$LOG_DIR/orchestrator.log"
  else
    echo "[obb] $(date -Is) FAIL  $label seed=$seed rc=$rc (no JSON line in $logf)" \
      | tee -a "$LOG_DIR/orchestrator.log"
  fi
}

# Alpha trial 23 config: c2,c5,w2,w3,w4 h=8 attn=none hw=False lam_a=0.0966 cap=100000
# OTC   trial 28 config: c2,c5,w2,w3,w4 h=4 attn=quaternion hw=True/0.137
#                        lam_a=1.48e-5 lam_attn=1.27e-3 cap=50000

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  run_cell "optuna_best_alpha" "$SEED" "bitcoin_alpha" 8  100000 \
    "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
    "HSIKAN_MAX_K3=100000" "HSIKAN_MAX_K2=100000" \
    "HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301"

  run_cell "optuna_best_otc" "$SEED" "bitcoin_otc"   4   50000 \
    "HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4" \
    "HSIKAN_MAX_K3=50000" "HSIKAN_MAX_K2=50000" \
    "HSIKAN_ATTENTION_M_E=quaternion" \
    "HSIKAN_ATTENTION_HIGHWAY=1" \
    "HSIKAN_ATTENTION_HIGHWAY_MAX=0.13682674286852775" \
    "HSIKAN_ALPHA_ENTROPY_LAMBDA=1.4777880758638605e-05" \
    "HSIKAN_ATTN_ENTROPY_LAMBDA=0.0012729880784274699"
done

echo "[obb] $(date -Is) DONE all 10 runs; jsonl=$RESULTS_FILE" \
  | tee -a "$LOG_DIR/orchestrator.log"

# Quick aggregate
python - <<PY
import json, statistics, pathlib
rows = [json.loads(l) for l in pathlib.Path("$RESULTS_FILE").read_text().splitlines() if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["run_label"], []).append(r["auc"])
for label, aucs in sorted(by.items()):
    if len(aucs) >= 2:
        print(f"{label}: n={len(aucs)} mean={statistics.mean(aucs):.4f} pstdev={statistics.pstdev(aucs):.4f}")
    else:
        print(f"{label}: n={len(aucs)} mean={statistics.mean(aucs):.4f}")
PY
