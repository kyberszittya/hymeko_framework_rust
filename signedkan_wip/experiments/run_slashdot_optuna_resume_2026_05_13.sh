#!/usr/bin/env bash
# Resume Slashdot Optuna on the canonical 010510Z stamp / DB.
# Waits for the in-flight HyMeYOLO redo (or any signedkan_wip GPU job) to
# clear, then launches 30 fresh trials on the same study
# (slashdot_20260513T010510Z). The 2 prior FAIL trials are already in the DB
# and don't block; Optuna's TPE state is preserved.
#
# Env matches the second Bitcoin/Slashdot relaunch that worked:
# HYMEKO_CYCLE_CACHE=1 + HSIKAN_TOPK_MODE=per_vertex + HSIKAN_TOPK_K=128 +
# HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION=1.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

LOG="signedkan_wip/experiments/results/optuna_slashdot_resume_20260513T010510Z.log"

echo "=== SLASHDOT RESUME $(date -u +%Y-%m-%dT%H:%M:%SZ) (waiting for HyMeYOLO) ===" >> "$LOG"

# Wait for any signedkan_wip GPU job to finish.
SELF_PID="$$"
SELF_NAME="$(basename "$0")"
while pgrep -af 'signedkan_wip\.src\.run_optuna_search|signedkan_wip\.src\.run_final_cell|signedkan_wip\.src\.vision\.train_circles_ricci' \
      | grep -vE "^${SELF_PID} " | grep -vF "$SELF_NAME" | grep -q .; do
  sleep 60
done
echo "=== GPU free $(date -u +%Y-%m-%dT%H:%M:%SZ); launching Slashdot Optuna ===" >> "$LOG"

export HSIKAN_CYCLE_BATCH="${HSIKAN_CYCLE_BATCH:-2000}"
export N_TRIALS="${N_TRIALS:-30}"
export N_EPOCHS="${N_EPOCHS:-80}"
export OPTUNA_STAMP="20260513T010510Z"
export OPTUNA_STORAGE="sqlite:///signedkan_wip/experiments/results/optuna_serial_${OPTUNA_STAMP}.db"
export OPTUNA_DATASETS="slashdot"
export HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION=1
export HYMEKO_CYCLE_CACHE=1
export HSIKAN_TOPK_MODE=per_vertex
export HSIKAN_TOPK_K=128

echo "git SHA: $(git rev-parse HEAD)" >> "$LOG"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> "$LOG"
echo "env: HYMEKO_CYCLE_CACHE=$HYMEKO_CYCLE_CACHE HSIKAN_TOPK_MODE=$HSIKAN_TOPK_MODE HSIKAN_TOPK_K=$HSIKAN_TOPK_K HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION=$HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION" >> "$LOG"
echo "cache files at launch: $(ls ~/.cache/hymeko/cycles_v1/ | wc -l) ($(du -sh ~/.cache/hymeko/cycles_v1/ | awk '{print $1}'))" >> "$LOG"
echo "----" >> "$LOG"

bash signedkan_wip/experiments/run_optuna_serial_datasets.sh >> "$LOG" 2>&1
echo "=== SLASHDOT RESUME DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
