#!/usr/bin/env bash
set -u
WT=/Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_coin_r9_wt
PY=/Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_framework_rust/.venv/bin/python3
cd "$WT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=reports/2026-08-06-r11-6c-composition
for off in 0 7 14 21 28 35 42 49; do
  "$PY" -m hymeko_rl.experiments.r11_6c_exact_zero_composition --phase eval --out "$OUT" \
    --offset "$off" --limit 7 > "$OUT/w_${off}.log" 2>&1 &
done
wait
echo "=== eval done; merging ==="
"$PY" -m hymeko_rl.experiments.r11_6c_exact_zero_composition --phase merge --out "$OUT" 2>&1 | tee "$OUT/merge.txt"
echo "R11_6C_ALL_DONE"
