#!/usr/bin/env bash
#
# HyMeYOLO `+ricci-mod` Ricci-feature scale sweep — 2026-05-16
#
# Sweeps the `--ricci-scale` multiplier over {0.05, 0.10, 0.20, 0.40,
# 0.80, 1.00} at the protocol-parity scale of the 2026-05-13 5-seed
# backfill: n_train=5000, epochs=50, lr=3e-3, +ricci-mod only.
#
# 6 scales × 5 seeds = 30 runs × ~10 min each ≈ 5 hours single-stream.
#
# Each run:
#   * cgroup-capped at 16 GB MemoryMax (per `feedback_ulimit_vs_cuda`).
#   * timeout 1800s (10 min) per run; sweep would burn 50 minutes if
#     all runs hit timeout, which they will not at the +ricci-mod
#     per-row wall of ~597s.
#   * jsonl row written under
#     signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_<STAMP>/
#   * stderr captured per run.
#
# Plan: docs/plans/2026-05-16-hymeyolo-ricci-weight-sweep/
# Report target: reports/2026-05-16-hymeyolo-ricci-weight-sweep.md
#
set -euo pipefail

REPO_ROOT="/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust"
cd "$REPO_ROOT"

# Use miniconda3 python (torch 2.11) for protocol parity with the
# 2026-05-13 5-seed backfill. The .venv has CORE-pinned torch 2.4.1
# but the prior 5-seed numbers we're comparing against were on
# miniconda. Mixing toolchains across the comparison would invalidate
# the paired-seed Δ.
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
PY=/home/kyberszittya/miniconda3/bin/python

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_${STAMP}"
mkdir -p "$OUT_DIR"
MASTER="$OUT_DIR/orchestrator.log"

GIT_SHA="$(git rev-parse HEAD)"
echo "[$(date -Is)] sweep start  STAMP=$STAMP  git=$GIT_SHA" | tee -a "$MASTER"
echo "[$(date -Is)] tool: $($PY -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())')" | tee -a "$MASTER"

SCALES=(0.05 0.10 0.20 0.40 0.80 1.00)
SEEDS=(0 1 2 3 4)
N_IMAGES=5000
N_EPOCHS=50
LR=0.003
PER_RUN_TIMEOUT=1800   # 30 min hard cap; +ricci-mod measures at ~600 s
                       # in the May 13 protocol.

run_one() {
  local scale="$1" seed="$2"
  # Scale slug for filenames (replace '.' with '_').
  local slug="s$(echo "$scale" | tr '.' '_')_seed${seed}"
  local jsonl_out="$OUT_DIR/${slug}.jsonl"
  local stdout_log="$OUT_DIR/${slug}.log"
  local stderr_log="$OUT_DIR/${slug}.err"
  local t0
  t0=$(date +%s)
  echo "[$(date -Is)] START scale=$scale seed=$seed" | tee -a "$MASTER"

  # Defensive opt-out from the 2026-05-16-evening defaults flip
  # (warm-start + cosine + warmup are now the canonical defaults). This
  # sweep was the honest-baseline measurement; preserving the original
  # protocol means explicitly NOT using either lever. Future re-runs
  # of this orchestrator reproduce the published 0.5041 ± 0.039 value.
  local cmd=("$PY" -m signedkan_wip.src.vision.train_circles_ricci
             --n-images "$N_IMAGES" --epochs "$N_EPOCHS" --lr "$LR"
             --seed "$seed" --ricci-scale "$scale"
             --no-warm-start
             --schedule constant
             --warmup-epochs 0
             --configs "+ricci-mod"
             --jsonl-out "$jsonl_out")

  local scope_name="hymeyolo-sweep-${slug}-${STAMP}.scope"

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
    echo "[$(date -Is)] OK    scale=$scale seed=$seed rc=$rc elapsed=${elapsed}s row=$row" \
      | tee -a "$MASTER"
  else
    echo "[$(date -Is)] FAIL  scale=$scale seed=$seed rc=$rc elapsed=${elapsed}s NO_JSONL" \
      | tee -a "$MASTER"
  fi
}

for scale in "${SCALES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "$scale" "$seed"
  done
done

echo "[$(date -Is)] sweep end  $(ls -1 "$OUT_DIR"/*.jsonl 2>/dev/null | wc -l)/30 rows" \
  | tee -a "$MASTER"
echo "Results: $OUT_DIR"
