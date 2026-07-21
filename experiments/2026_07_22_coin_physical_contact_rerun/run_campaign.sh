#!/usr/bin/env bash
# Coin physical-contact SAC-vs-TD3 rerun campaign (2026-07-22) — LOCAL Mac, no katolab.
# 5 seeds x {SAC,TD3} x 100k steps from the SAME BC zero-residual init under corrected physics.
# Modest parallelism (MAXP concurrent) to respect the 16 GB RSS cap; host-local out dirs under the artifact root.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. PYTHONUNBUFFERED=1

AR=experiments/2026_07_22_coin_physical_contact_rerun/campaign
SEEDS=(0 1 2 3 4)
ALGOS=(SAC TD3)
STEPS=100000
EVAL_EVERY=10000
MAXP=4                              # concurrent runs (each ~single-digit % RSS; 4 stays well under 16 GB)

mkdir -p "$AR"
# bash-3.2-safe batching: launch up to MAXP, barrier-wait the batch, repeat.
running=0
for algo in "${ALGOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    lo=$(echo "$algo" | tr '[:upper:]' '[:lower:]')
    out="$AR/${lo}_s${seed}"
    log="$AR/${lo}_s${seed}.log"
    echo "[launch] $algo seed=$seed -> $out"
    python -m hymeko_rl.experiments.coin_physical_contact_rerun \
        --algo "$algo" --seed "$seed" --steps "$STEPS" --eval-every "$EVAL_EVERY" --out "$out" \
        > "$log" 2>&1 &
    running=$((running+1))
    if (( running % MAXP == 0 )); then wait; fi
  done
done
wait
echo "[campaign] all 10 runs complete -> $AR"
