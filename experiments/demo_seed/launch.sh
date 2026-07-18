#!/bin/bash
# Launch all 16 demo-seed cells (2 arms x 8 seeds) with a concurrency cap of 8, 2 torch threads each.
# Interleaves cold/demo_seed per seed so both arms progress if interrupted. Logs per cell to logs/.
set -u
cd "$(dirname "$0")"
export DEMO_SEED_THREADS=2
export PYTHONPATH="$PWD:$(cd ../.. && pwd)"
PY="$(cd ../.. && pwd)/.venv/bin/python"
mkdir -p logs
CAP=8

run_cell() { "$PY" exp_demo_seed.py "$1" "$2" >"logs/${1}_seed${2}.log" 2>&1; }

for s in 0 1 2 3 4 5 6 7; do
  for arm in cold demo_seed; do
    while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 2; done  # bash 3.2: no `wait -n`, poll instead
    run_cell "$arm" "$s" &
    echo "[launch] started $arm seed$s (pid $!)"
  done
done
wait
echo "[launch] ALL 16 CELLS DONE"
