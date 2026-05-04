#!/usr/bin/env bash
# Run all cells of the final results table, each in a fresh subprocess
# to avoid cudagraph cache pollution between widths/datasets.
set -euo pipefail
export HSIKAN_TORCH_COMPILE=1

OUT="signedkan_wip/experiments/results/final_table.jsonl"
mkdir -p "$(dirname "$OUT")"
> "$OUT"

run_cell() {
  local args="$*"
  echo "  running: $args" >&2
  local result
  if result=$(python3 -m signedkan_wip.src.run_final_cell $args 2>/dev/null | tail -1); then
    if [[ -n "$result" && "$result" != "null" ]]; then
      echo "$result" >> "$OUT"
      echo "$result" | python3 -c "import sys, json; d=json.loads(sys.stdin.read()); print(f'    -> {d}')" >&2
    fi
  fi
}

# --- Edge-sign prediction (HSiKAN mixed-arity + SGCN) ---
for ds in bitcoin_alpha bitcoin_otc; do
  for h in 16 8 4; do
    run_cell --dataset $ds --model HSiKAN --hidden $h --n-epochs 100
  done
  run_cell --dataset $ds --model SGCN --hidden 32 --n-epochs 100
done

# Slashdot — bigger budget, mixed arity, cycle-batched
export HSIKAN_COMPILE_MODE=default     # cycle batching breaks cudagraphs
for h in 16 8 4; do
  run_cell --dataset slashdot --model HSiKAN --hidden $h --n-epochs 60 \
            --max-k4 200000
done
run_cell --dataset slashdot --model SGCN --hidden 32 --n-epochs 60
unset HSIKAN_COMPILE_MODE

# SBM (synthetic block-model signed)
for ds in sbm_n200 sbm_n400; do
  for h in 16 8 4; do
    run_cell --dataset $ds --model HSiKAN --hidden $h --n-epochs 200
  done
  run_cell --dataset $ds --model SGCN --hidden 32 --n-epochs 200
done

# Scene-graph (k=2 fallback, HSiKAN-only)
for h in 16 8 4; do
  run_cell --dataset scene --model HSiKAN --hidden $h --n-epochs 80
done

# Kinematic + Pose (graph-level, HSiKAN-only)
for arity in 4 6; do
  for h in 16 8 4; do
    run_cell --dataset kinematic_k$arity --model HSiKAN --hidden $h \
              --n-epochs 30
    run_cell --dataset pose_k$arity --model HSiKAN --hidden $h \
              --n-epochs 100
  done
done

echo "DONE — wrote $OUT" >&2
wc -l "$OUT"
