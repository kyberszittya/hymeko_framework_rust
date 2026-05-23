#!/usr/bin/env bash
# Phase-6 — Universality test for three new regulariser arms.
#
# Tests whether the three "path toward universal entropy cost" designs
# preserve the 6 significant positives and neutralise the 4 significant
# negatives observed in the 60-experiment phase 1-5 corpus.
#
#   Path B — scalar_entropy_normalized   L = λ · H(A) / log2(rank(A))
#   Path A — entropy_target              L = λ · (H_norm - H*)²
#   Path C — structural_composite        L = λ · max(H_norm, σ_max/2, 1 - sr/rank)
#
# Four probe datasets chosen to span the phase-1-5 result surface:
#   mnist_small      — existing significant positive (anchor)
#   mnist_capsnet    — existing significant NEGATIVE (stress test)
#   spirals          — existing strong positive (sanity check)
#   circles          — existing significant NEGATIVE (second stress test)
#
# Dataflow view only to fit the overnight budget; factor view can
# follow in phase 6b if signals warrant.

set -u
LOG=/tmp/thesis_iv_views_ph6.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH6 suite started: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG

RUN() {
  local name="$1"; shift
  echo "" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] START: $name" | tee -a $LOG
  echo "  cmd: $*" | tee -a $LOG
  python3 python/benches/thesis_iv_hard/run_benchmark.py "$@" >> $LOG 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE:  $name" | tee -a $LOG \
    || echo "[$(date +%H:%M:%S)] FAIL:  $name (exit=$?)" | tee -a $LOG
}

# ═══════════════════════════════════════════════════════════════════
# Spirals / circles first — cheap, early results to confirm arms work
# ═══════════════════════════════════════════════════════════════════

for arm in scalar_entropy_normalized entropy_target structural_composite; do
  RUN "spirals ${arm} dataflow 100-seed × 50 epochs" \
    --datasets spirals \
    --arms baseline ${arm} \
    --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
  RUN "circles ${arm} dataflow 100-seed × 50 epochs" \
    --datasets circles \
    --arms baseline ${arm} \
    --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# ═══════════════════════════════════════════════════════════════════
# MNIST plain MLP — publication power (33×15), matches the anchor
# ═══════════════════════════════════════════════════════════════════

for arm in scalar_entropy_normalized entropy_target structural_composite; do
  RUN "mnist_small ${arm} dataflow 33-seed × 15 epochs" \
    --datasets mnist_small \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# ═══════════════════════════════════════════════════════════════════
# CapsMLP MNIST — the phase-5 significant negative, 33×10
# ═══════════════════════════════════════════════════════════════════

for arm in scalar_entropy_normalized entropy_target structural_composite; do
  RUN "mnist_capsnet ${arm} dataflow 33-seed × 10 epochs" \
    --datasets mnist_capsnet \
    --arms baseline ${arm} \
    --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH6 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
