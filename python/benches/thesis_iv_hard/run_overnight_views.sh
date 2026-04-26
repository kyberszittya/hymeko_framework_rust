#!/usr/bin/env bash
# Overnight benchmark suite #2 — dataflow vs factor view × convergence speed.
#
# Builds on the previous overnight suite's findings:
#   - Plain MLP scalar_entropy is the anchor positive result
#   - ResMLP-20 KL-trajectory hinted at positive at 15 seeds (unreplicated at 33)
#   - CIFAR-10 was null at 10 epochs × narrow arches
#
# This suite adds two new dimensions:
#   1. **Factor-view adjacency** (thesis §6.1.2, clique expansion) alongside
#      dataflow-view. Tests whether the view choice matters empirically.
#   2. **Per-epoch val accuracy tracking** to measure convergence speed,
#      not just final accuracy. Crucial because the scalar_entropy effect
#      shrinks with training budget — the regularizer may simply be an
#      early-training accelerant.
#
# Total estimated budget: ~6 hours on RTX 2070 Super.

set -u
LOG=/tmp/thesis_iv_views.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views suite started: $(date)" | tee -a $LOG
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
# EXPERIMENT 1 — Dataflow vs factor view on plain MLP, publication-power
# ═══════════════════════════════════════════════════════════════════
# 33 seeds × 15 epochs at our established best configuration.
# Directly comparable to the 33-seed dataflow result from last night
# (+0.149%, t=+2.88). Answers: does factor view give equal, stronger,
# or different signal from dataflow?

RUN "Plain MLP factor-view 33-seed × 15 epochs" \
  --datasets mnist_small \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view factor

RUN "Plain MLP factor-view KL trajectory 33-seed × 5 epochs" \
  --datasets mnist_small \
  --arms baseline kl_trajectory \
  --seeds 33 --epochs 5 --lam 10.0 --reg-every-n 100 --view factor

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — ResMLP-20 view × arm matrix, cements KL finding
# ═══════════════════════════════════════════════════════════════════
# ResMLP-20 showed unreplicated positive for KL at 15 seeds × 5 epochs.
# Run at 33 seeds × 5 epochs for BOTH views to cement or refute.

RUN "ResMLP-20 KL-trajectory dataflow 33-seed × 5 epochs" \
  --datasets mnist_resnet_20 \
  --arms baseline kl_trajectory \
  --seeds 33 --epochs 5 --lam 10.0 --reg-every-n 100 --view dataflow

RUN "ResMLP-20 scalar_entropy factor 33-seed × 5 epochs" \
  --datasets mnist_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 5 --lam 0.1 --reg-every-n 10 --view factor

RUN "ResMLP-20 KL-trajectory factor 33-seed × 5 epochs" \
  --datasets mnist_resnet_20 \
  --arms baseline kl_trajectory \
  --seeds 33 --epochs 5 --lam 10.0 --reg-every-n 100 --view factor

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Convergence speed on plain MLP (30-epoch curves)
# ═══════════════════════════════════════════════════════════════════
# Per-epoch tracking is now enabled. 30 epochs gives us a full
# learning curve for each seed. Key question: does the regularizer
# accelerate convergence even when asymptotic accuracies are similar?

RUN "Plain MLP convergence curves, dataflow 15-seed × 30 epochs" \
  --datasets mnist_small \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 30 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "Plain MLP convergence curves, factor 15-seed × 30 epochs" \
  --datasets mnist_small \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 30 --lam 0.1 --reg-every-n 10 --view factor

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — CIFAR-10 view comparison (the outstanding question)
# ═══════════════════════════════════════════════════════════════════
# Previous CIFAR-10 runs with dataflow view showed null. Does factor
# view give a different result on CIFAR? Longer training (20 epochs)
# to address the "undertrained baseline" hypothesis.

RUN "CIFAR-10 plain MLP dataflow 15-seed × 20 epochs" \
  --datasets cifar10 \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "CIFAR-10 plain MLP factor 15-seed × 20 epochs" \
  --datasets cifar10 \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 --view factor

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
