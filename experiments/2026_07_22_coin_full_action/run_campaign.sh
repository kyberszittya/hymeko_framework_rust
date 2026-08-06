#!/usr/bin/env bash
# Standalone full-action SAC-vs-TD3 campaign (2026-07-22) — LOCAL Mac.
# 5 seeds x {SAC,TD3} x 100k, each from the SAME competent BC (script disabled). bash-3.2-safe batching.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_framework_rust/.venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. PYTHONUNBUFFERED=1

AR=experiments/2026_07_22_coin_full_action/campaign
SEEDS=(0 1 2 3 4); ALGOS=(SAC TD3); STEPS=100000; EVAL_EVERY=10000; MAXP=4
mkdir -p "$AR"; running=0
for algo in "${ALGOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    lo=$(echo "$algo" | tr '[:upper:]' '[:lower:]')
    python -m hymeko_rl.experiments.coin_full_action_rl --algo "$algo" --seed "$seed" \
        --steps "$STEPS" --eval-every "$EVAL_EVERY" --out "$AR/${lo}_s${seed}" > "$AR/${lo}_s${seed}.log" 2>&1 &
    running=$((running+1)); (( running % MAXP == 0 )) && wait
  done
done
wait
echo "[campaign] all 10 full-action runs complete -> $AR"
