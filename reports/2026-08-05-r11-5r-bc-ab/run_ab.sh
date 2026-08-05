#!/usr/bin/env bash
# R11.5R BC B0-vs-B1 A/B: same r11_4b_conditioned_bc harness, only the dataset (target theta) differs.
#   B0 = nominal R11.4B dataset ; B1 = robust-recertified dataset (dataset_b1).
set -u
WT=/Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_coin_r9_wt
PY=/Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_framework_rust/.venv/bin/python3
cd "$WT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

AB=reports/2026-08-05-r11-5r-bc-ab
B0_DS=reports/2026-08-03-r11-4b-bc/dataset
B1_DS=reports/2026-08-05-r11-5r-robust-teacher/dataset_b1
STEP=7

launch_eval() {  # $1=dataset_dir  $2=eval_dir
  rm -rf "$2"; mkdir -p "$2"
  for off in 0 7 14 21 28 35 42 49; do
    "$PY" -m hymeko_rl.experiments.r11_4b_conditioned_bc --phase eval \
      --dataset-dir "$1" --eval-dir "$2" --offset "$off" --limit "$STEP" \
      > "$2/w_${off}.log" 2>&1 &
  done
}

echo "launching B0 (nominal) + B1 (robust) eval fanout, 8+8 workers"
launch_eval "$B0_DS" "$AB/eval_b0"
launch_eval "$B1_DS" "$AB/eval_b1"
wait
echo "=== eval done; gating ==="
for arm in b0 b1; do
  echo "----- $arm -----"
  "$PY" -m hymeko_rl.experiments.r11_4b_conditioned_bc --phase gate --eval-dir "$AB/eval_$arm" \
    2>&1 | tee "$AB/gate_$arm.txt"
done
echo "R11_5R_BC_AB_DONE"
