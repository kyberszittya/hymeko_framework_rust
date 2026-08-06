#!/usr/bin/env bash
# Forward-walking footstep-policy training (CEM over the WBC/DCM footstep env) — CPU-bound, scales with cores.
# Run from the humanoid worktree root (the branch with scenarios/humanoid/footstep_env.py), e.g. on kato14/15:
#   HYMEKO_WORKERS=32 HYMEKO_ITERS=120 bash scripts/kato15/footstep_walk_run.sh
set -euo pipefail
ROOT="${HYMEKO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"     # worktree root (two dirs up from this script)
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HYMEKO_WORKERS="${HYMEKO_WORKERS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
export HYMEKO_ITERS="${HYMEKO_ITERS:-120}"
export HYMEKO_POP="${HYMEKO_POP:-40}"
export HYMEKO_W_FORWARD="${HYMEKO_W_FORWARD:-45}"
export HYMEKO_FWD_STRIDE="${HYMEKO_FWD_STRIDE:-0.06}"
export HYMEKO_OUT="${HYMEKO_OUT:-experiments/2026_07_29_humanoid_fwalk}"
PY="${HYMEKO_PY:-.venv/bin/python}"; [ -x "$PY" ] || PY="python"
mkdir -p "$HYMEKO_OUT"
setsid nohup "$PY" -m scenarios.humanoid.train_footstep_walk \
  --iters "$HYMEKO_ITERS" --pop "$HYMEKO_POP" --workers "$HYMEKO_WORKERS" \
  --w_forward "$HYMEKO_W_FORWARD" --fwd_stride "$HYMEKO_FWD_STRIDE" --out "$HYMEKO_OUT" \
  > "$HYMEKO_OUT/fwalk.log" 2>&1 < /dev/null &
echo "footstep-walk training launched pid=$! workers=$HYMEKO_WORKERS iters=$HYMEKO_ITERS -> $HYMEKO_OUT/fwalk.log"
echo "when done: python -m scenarios.humanoid.train_footstep_walk --render --out $HYMEKO_OUT"
