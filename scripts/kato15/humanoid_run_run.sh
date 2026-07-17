#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hymeko_framework_rust"
export HYMEKO_DEVICE=cuda MUJOCO_GL=egl PYTHONPATH="$HOME/hymeko_framework_rust"
export TORCHINDUCTOR_CACHE_DIR="$HOME/hymeko_framework_rust/.torchinductor_cache"
export HYMEKO_OUT="experiments/2026_07_17_humanoid_run"
export HYMEKO_TARGET="${HYMEKO_TARGET:-3.0}" HYMEKO_STEPS="${HYMEKO_STEPS:-1500000}" HYMEKO_SEEDS="${HYMEKO_SEEDS:-0,1,2}"
mkdir -p "$HYMEKO_OUT"
setsid nohup .venv_stand/bin/python "$HOME/hymeko_framework_rust/scripts/kato15/humanoid_run_launch.py" \
  > "$HYMEKO_OUT/hrun.log" 2>&1 < /dev/null &
echo "humanoid-run launched pid=$! -> $HYMEKO_OUT/hrun.log"
