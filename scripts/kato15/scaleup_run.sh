#!/usr/bin/env bash
# Humanoid scale-up. Run ON the free box. Seed-split example: kato15 seeds 0-2, kato14 seeds 3-4.
#   HYMEKO_SEEDS=0,1,2 bash ~/scaleup_run.sh        # on kato15
#   HYMEKO_SEEDS=3,4   bash ~/scaleup_run.sh        # on kato14
set -euo pipefail
cd "$HOME/hymeko_framework_rust"
HOST=$(hostname)
export HYMEKO_DEVICE=cuda MUJOCO_GL=egl PYTHONPATH="$HOME/hymeko_framework_rust"
export TORCHINDUCTOR_CACHE_DIR="$HOME/hymeko_framework_rust/.torchinductor_cache"
export HYMEKO_OUT="experiments/2026_07_17_humanoid_scaleup_${HOST}"     # per-box dir (NFS shared; avoid concurrent writes)
export HYMEKO_SEEDS="${HYMEKO_SEEDS:-0,1,2,3,4}"
mkdir -p "$HYMEKO_OUT"
setsid nohup .venv_stand/bin/python scaleup_launch.py > "$HYMEKO_OUT/scaleup.log" 2>&1 < /dev/null &
echo "scaleup launched on $HOST pid=$! seeds=$HYMEKO_SEEDS -> $HYMEKO_OUT/scaleup.log"
