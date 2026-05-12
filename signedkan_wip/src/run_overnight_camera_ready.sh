#!/usr/bin/env bash
# Overnight camera-ready benchmark sweep.
#
# Runs each cell in an isolated CUDA process (fresh cudagraph cache)
# and aggregates into one JSONL.  Total wall-clock estimate:
#   (1) Bitcoin/SBM 5-seed × 2 widths × 4 datasets ≈ 40 cells × ~45s
#   (2) Slashdot 5-seed × h=4 × max_k4=500000 ≈ 5 cells × ~10min
# Roughly 80-90 min.
#
# Usage:
#   bash signedkan_wip/src/run_overnight_camera_ready.sh
#
# Output:
#   signedkan_wip/experiments/results/overnight_camera_ready.jsonl
set -euo pipefail
export HSIKAN_TORCH_COMPILE=1

OUT_FILE="signedkan_wip/experiments/results/overnight_camera_ready.jsonl"
LOG_FILE="signedkan_wip/experiments/results/overnight_camera_ready.log"
mkdir -p "$(dirname "$OUT_FILE")"
> "$OUT_FILE"
> "$LOG_FILE"

run_cell() {
  local description="$1"; shift
  local args="$*"
  local stamp
  stamp=$(date +"%H:%M:%S")
  echo "[$stamp] $description :: $args" | tee -a "$LOG_FILE"
  local result
  if result=$(timeout 7200 python3 -m signedkan_wip.src.run_final_cell $args 2>&1 | tail -1); then
    if [[ -n "$result" && "$result" != "null" && "$result" =~ ^\{ ]]; then
      echo "$result" >> "$OUT_FILE"
      python3 -c "
import sys, json
d = json.loads(sys.argv[1])
auc = d.get('auc')
f1m = d.get('f1m', 0)
lat = d.get('fwd_per_call_ms', 0)
hidden = d.get('hidden')
seed = d.get('seed', 0)
ds = d.get('dataset')
print(f'    -> {ds} h={hidden} seed={seed}  AUC={auc:.4f}  F1m={f1m:.4f}  lat={lat:.1f}ms')
" "$result" | tee -a "$LOG_FILE"
    else
      echo "    !! cell failed; raw output:" | tee -a "$LOG_FILE"
      echo "    $result" | head -3 | tee -a "$LOG_FILE"
    fi
  fi
}

echo "==========================================================="
echo " Phase 1 — Bitcoin / SBM 5-seed sweep"
echo "==========================================================="
# Bitcoin Alpha: h=16 (paper baseline) + h=4 (pruning point), 5 seeds
for seed in 0 1 2 3 4; do
  for h in 16 4; do
    run_cell "bitcoin_alpha h=$h seed=$seed" \
      --dataset bitcoin_alpha --model HSiKAN --hidden $h \
      --n-epochs 100 --seed $seed
  done
done

# Bitcoin OTC: h=16 + h=8 (Table I best), 5 seeds
for seed in 0 1 2 3 4; do
  for h in 16 8; do
    run_cell "bitcoin_otc h=$h seed=$seed" \
      --dataset bitcoin_otc --model HSiKAN --hidden $h \
      --n-epochs 100 --seed $seed
  done
done

# SBM n=200: h=16 + h=8 (Table I best), 5 seeds
for seed in 0 1 2 3 4; do
  for h in 16 8; do
    run_cell "sbm_n200 h=$h seed=$seed" \
      --dataset sbm_n200 --model HSiKAN --hidden $h \
      --n-epochs 200 --seed $seed
  done
done

# SBM n=400: h=16 + h=4 (Table I best), 5 seeds
for seed in 0 1 2 3 4; do
  for h in 16 4; do
    run_cell "sbm_n400 h=$h seed=$seed" \
      --dataset sbm_n400 --model HSiKAN --hidden $h \
      --n-epochs 200 --seed $seed
  done
done

# SGCN baselines per dataset, 5 seeds (single h=32)
for ds in bitcoin_alpha bitcoin_otc sbm_n200 sbm_n400; do
  for seed in 0 1 2 3 4; do
    n_ep=100
    if [[ "$ds" == sbm_n* ]]; then n_ep=200; fi
    run_cell "$ds SGCN seed=$seed" \
      --dataset $ds --model SGCN --hidden 32 \
      --n-epochs $n_ep --seed $seed
  done
done

echo "==========================================================="
echo " Phase 2 — Slashdot 5-seed at SOTA pruning point (h=4, 500k cycles)"
echo "==========================================================="
export HSIKAN_COMPILE_MODE=default   # cycle-batched mode
export HSIKAN_ARITIES=2,3,4,5
for seed in 0 1 2 3 4; do
  run_cell "slashdot h=4 SOTA seed=$seed" \
    --dataset slashdot --model HSiKAN --hidden 4 \
    --n-epochs 80 --max-k4 500000 --seed $seed
done
# Also one 5-seed h=16 reference for the strict-pruning comparison
for seed in 0 1 2 3 4; do
  run_cell "slashdot h=16 ref seed=$seed" \
    --dataset slashdot --model HSiKAN --hidden 16 \
    --n-epochs 80 --max-k4 500000 --seed $seed
done
unset HSIKAN_ARITIES HSIKAN_COMPILE_MODE

# SGCN-Slashdot 5-seed for apples-to-apples
for seed in 0 1 2 3 4; do
  run_cell "slashdot SGCN seed=$seed" \
    --dataset slashdot --model SGCN --hidden 32 \
    --n-epochs 60 --seed $seed
done

echo "==========================================================="
echo " DONE  -- $(wc -l < "$OUT_FILE") cells written to $OUT_FILE"
date | tee -a "$LOG_FILE"
echo "==========================================================="
