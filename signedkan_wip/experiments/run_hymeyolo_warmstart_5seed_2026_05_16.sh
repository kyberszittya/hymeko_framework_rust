#!/usr/bin/env bash
#
# HyMeYOLO Stage-A-1: warm-start query corners — 5-seed validation.
# 2026-05-16
#
# Runs `+ricci-mod` with --warm-start on, 5 seeds × 1 ricci-scale,
# same protocol as the morning ricci-scale sweep (n=5000, e=50,
# lr=3e-3, miniconda3 / torch 2.11, cgroup 16 GB).
#
# The ricci-scale used here is taken from the sweep's winner if one
# exists; otherwise default 1.0. Pass the scale as the first
# argument; default is 1.0.
#
# DO NOT launch this script until the ricci-scale sweep finishes:
#   signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_20260516T002116Z/
#
# Plan: docs/plans/2026-05-16-hymeyolo-warmstart-query-init/
# Report target: reports/2026-05-16-hymeyolo-warmstart-5seed.md
#
set -euo pipefail

REPO_ROOT="/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust"
cd "$REPO_ROOT"

# Protocol parity with the morning sweep.
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
PY=/home/kyberszittya/miniconda3/bin/python

RICCI_SCALE="${1:-1.0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/signedkan_wip/experiments/results/hymeyolo_warmstart_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
MASTER="$OUT_DIR/orchestrator.log"

GIT_SHA="$(git rev-parse HEAD)"
echo "[$(date -Is)] warmstart 5-seed start  STAMP=$STAMP  git=$GIT_SHA  ricci_scale=$RICCI_SCALE" \
  | tee -a "$MASTER"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  2>&1 | tee -a "$MASTER"

SEEDS=(0 1 2 3 4)
N_IMAGES=5000
N_EPOCHS=50
LR=0.003
PER_RUN_TIMEOUT=1800

run_one() {
  local seed="$1"
  local slug="warmstart_seed${seed}_scale$(echo "$RICCI_SCALE" | tr '.' '_')"
  local jsonl_out="$OUT_DIR/${slug}.jsonl"
  local stdout_log="$OUT_DIR/${slug}.log"
  local stderr_log="$OUT_DIR/${slug}.err"
  local t0
  t0=$(date +%s)
  echo "[$(date -Is)] START seed=$seed scale=$RICCI_SCALE" | tee -a "$MASTER"

  # Defensive opt-out from the 2026-05-16-evening defaults flip on
  # --schedule (now defaults to cosine). This orchestrator measured
  # Stage A-1 (warm-start ONLY, constant LR, e=50) and is the
  # baseline for Stage A-2's paired comparison; preserving it means
  # explicitly forcing the LR schedule back to constant.
  # --warm-start is now the default but kept explicit here for
  # documentation clarity (the script's intent is to test it).
  local cmd=("$PY" -m signedkan_wip.src.vision.train_circles_ricci
             --n-images "$N_IMAGES" --epochs "$N_EPOCHS" --lr "$LR"
             --seed "$seed" --ricci-scale "$RICCI_SCALE"
             --warm-start
             --schedule constant
             --warmup-epochs 0
             --configs "+ricci-mod"
             --jsonl-out "$jsonl_out")

  local scope_name="hymeyolo-warmstart-${slug}-${STAMP}.scope"
  systemd-run --user --scope --quiet \
    --unit="$scope_name" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    timeout "$PER_RUN_TIMEOUT" "${cmd[@]}" \
    > "$stdout_log" 2> "$stderr_log" || true
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ -s "$jsonl_out" ]; then
    local row
    row=$(tail -1 "$jsonl_out")
    echo "[$(date -Is)] OK    seed=$seed rc=$rc elapsed=${elapsed}s row=$row" \
      | tee -a "$MASTER"
  else
    echo "[$(date -Is)] FAIL  seed=$seed rc=$rc elapsed=${elapsed}s NO_JSONL" \
      | tee -a "$MASTER"
  fi
}

for seed in "${SEEDS[@]}"; do
  run_one "$seed"
done

echo "[$(date -Is)] warmstart 5-seed end  $(ls -1 "$OUT_DIR"/*.jsonl 2>/dev/null | wc -l)/5 rows" \
  | tee -a "$MASTER"
echo "Results: $OUT_DIR"
