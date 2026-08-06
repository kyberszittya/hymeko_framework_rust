#!/usr/bin/env bash
# kato15 launch — SAC-from-scratch walking campaign (Aibo goal-reach / humanoid / cheetah × {flat,structural}
# × 5 seeds; pure SAC, no warm-start — the local A/B proved warm-start TRAPS). One entrypoint, mode arg.
#
# Prereqs (per reports/2026-07-07-session-handoff-aibo-standing-dagger.md):
#   * repo at ~/hymeko_framework_rust (sync the new files first — see SYNC below)
#   * the .venv_stand uv venv: torch 2.11.0+cu128 for kato15's CUDA 12.8. The repo CORE pin is cu132, whose
#     GPU build does NOT run on kato15 — do NOT use the default .venv on the GPU (§3 RL carve-out, venv-only).
#   * RTX 6000 Ada 48 GB (~1135 steps/s). 16 GB RSS cap via cgroups v2 (§4).
#
# SYNC from the Mac (run there, when kato15 is reachable) — includes train/sac.py (2026-07-17 compiled update):
#   tar czf - hymeko_rl/train/sac.py hymeko_rl/experiments/exp_sac_walk_campaign.py \
#     hymeko_rl/experiments/exp_sac_walk_validation.py hymeko_rl/experiments/exp_aibo_cip_walk.py \
#     hymeko_rl/experiments/exp_cip_verification_campaign.py hymeko_rl/env/locomotion_env.py \
#     hymeko_rl/env/locomotion_experts.py hymeko_rl/env/reward.py hymeko_rl/env/terrain.py \
#     hymeko_rl/env/track_gen.py hymeko_rl/tests/test_sac_compiled_update.py hymeko_rl/tests/test_aibo_walk.py \
#     data/robotics/*.hymeko scripts/kato15/run_sac_walk.sh \
#   | ssh kato15 'bash -lc "cd ~/hymeko_framework_rust && tar xzf - && find hymeko_rl -name \"*.pyc\" -delete"'
#
# RUN on kato15:
#   bash scripts/kato15/run_sac_walk.sh smoke        # 20k, Aibo flat, bounce {3,8} — path check
#   bash scripts/kato15/run_sac_walk.sh gpu-smoke    # 20k structural, compile vs eager — MEASURE the speedup
#   bash scripts/kato15/run_sac_walk.sh bounce-ab    # THE grid: {flat,structural}×{bounce 3,8}×tall bodies×5 seeds×800k
#   bash scripts/kato15/run_sac_walk.sh full         # original 30-cell reproduction
set -euo pipefail
cd "$HOME/hymeko_framework_rust"
MODE="${1:-full}"
export HYMEKO_DEVICE=cuda
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # headless GL (no rendering here, but keep it safe)

PY="$HOME/hymeko_framework_rust/.venv_stand/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY missing — create the .venv_stand (torch 2.11+cu128) per the handoff"; exit 1; }

case "$MODE" in
  smoke)     ARGS="--smoke" ;;                                            # 20k Aibo flat, bounce {3,8}
  gpu-smoke) ARGS="--bounce-ab --with-structural --seeds 1 --steps 20000" ;;  # 1 structural cell, compiled — speedup check
  bounce-ab) ARGS="--bounce-ab --with-structural" ;;                     # the full compiled grid (800k)
  full)      ARGS="" ;;                                                  # original 30-cell reproduction
  *) echo "ERROR: unknown mode '$MODE' (smoke|gpu-smoke|bounce-ab|full)"; exit 1 ;;
esac
OUT="experiments/2026_07_16_sac_walk_campaign"; mkdir -p "$OUT"
LOG="$OUT/kato15_${MODE}.log"
echo "[run_sac_walk] mode=$MODE args='$ARGS' device=cuda py=$PY -> $LOG"

# §3 GPU checks: no other workload; §4 16 GB RSS cap (cgroups v2 RSS gate, NOT ulimit -v).
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true
systemd-run --user --scope -p MemoryMax=16G \
  "$PY" -m hymeko_rl.experiments.exp_sac_walk_campaign $ARGS 2>&1 | tee "$LOG"
echo "[run_sac_walk] done -> $OUT/summary.json  (per-body flat-vs-structural dx + CIP propel-edge medians)"
