#!/usr/bin/env bash
#
# HyMeYOLO Stage A-2: warm-start + cosine LR + warmup + e=100 — 5-seed.
# 2026-05-16
#
# Builds on Stage A-1 (warm-start, paired Δ = +0.124, n=5/n=5).
# Probes whether the cosine-LR + longer-training lever lands the
# predicted +0.04 mAP_50 lift on top of warm-start.
#
# Protocol parity with Stage A-1 (miniconda3 torch 2.11, n_train=5000,
# lr=3e-3 peak). Differs only in:
#   * --epochs 100 (vs 50)
#   * --schedule cosine
#   * --warmup-epochs 10 (10% of epochs)
#   * --min-lr-ratio 0.01
#
# Plan: docs/plans/2026-05-16-hymeyolo-stage-a2-cosine/
# Report target: reports/2026-05-16-hymeyolo-stage-a2-5seed.md
#
set -euo pipefail

REPO_ROOT="/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust"
cd "$REPO_ROOT"

export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
PY=/home/kyberszittya/miniconda3/bin/python

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/signedkan_wip/experiments/results/hymeyolo_stage_a2_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
MASTER="$OUT_DIR/orchestrator.log"

GIT_SHA="$(git rev-parse HEAD)"
echo "[$(date -Is)] stage A-2 5-seed start  STAMP=$STAMP  git=$GIT_SHA" \
  | tee -a "$MASTER"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  2>&1 | tee -a "$MASTER"

SEEDS=(0 1 2 3 4)
N_IMAGES=5000
N_EPOCHS=100
LR=0.003
WARMUP=10
MIN_LR_RATIO=0.01
PER_RUN_TIMEOUT=2400   # 40 min cap; per-seed measures at ~19 min.

run_one() {
  local seed="$1"
  local slug="seed${seed}_e${N_EPOCHS}_cosine_warmup${WARMUP}"
  local jsonl_out="$OUT_DIR/${slug}.jsonl"
  local stdout_log="$OUT_DIR/${slug}.log"
  local stderr_log="$OUT_DIR/${slug}.err"
  local t0
  t0=$(date +%s)
  echo "[$(date -Is)] START seed=$seed" | tee -a "$MASTER"

  local cmd=("$PY" -m signedkan_wip.src.vision.train_circles_ricci
             --n-images "$N_IMAGES" --epochs "$N_EPOCHS" --lr "$LR"
             --seed "$seed" --ricci-scale 1.0
             --warm-start
             --schedule cosine
             --warmup-epochs "$WARMUP"
             --min-lr-ratio "$MIN_LR_RATIO"
             --configs "+ricci-mod"
             --jsonl-out "$jsonl_out")

  local scope_name="hymeyolo-stage-a2-${slug}-${STAMP}.scope"
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

echo "[$(date -Is)] stage A-2 5-seed end  $(ls -1 "$OUT_DIR"/*.jsonl 2>/dev/null | wc -l)/5 rows" \
  | tee -a "$MASTER"
echo "Results: $OUT_DIR"
