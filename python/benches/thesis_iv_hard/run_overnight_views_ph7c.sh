#!/usr/bin/env bash
# Phase-7c — Universality breadth test.
#
# Re-runs the two strongest universality candidates
#   - scalar_entropy_normalized   (Path B)
#   - entropy_target H*=0.5       (Path A)
# on the six datasets/architectures that were tested ONLY with the
# original scalar_entropy in phases 1-5:
#
#   mnist_resnet_20   (deep skip, original Δ=+0.046–0.070 marginal)
#   mnist_highway_20  (deep gated, original was untested at full power)
#   fashion_mnist     (image sibling, original null)
#   kmnist            (Japanese sibling, original null)
#   emnist_letters    (26-class sibling, directional + with σ drop)
#   svhn              (32×32 colour, original directionally negative)
#
# Goal: extend the universality claim from 4 stress datasets (phase 6/7)
# to 4 architectures + 4 image siblings + 1 deep-gated architecture.
#
# Auto-fires after Views PH7b suite finishes.
#
# Estimated wall-clock: ~17 hours.

set -u
LOG=/tmp/thesis_iv_views_ph7c.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH7c suite started: $(date)" | tee -a $LOG
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
# Lighter runs first (deep arches at 5 epochs are quick per seed)
# ═══════════════════════════════════════════════════════════════════
for arm in scalar_entropy_normalized entropy_target; do
  RUN "ResMLP-20 ${arm} dataflow 33-seed × 5 epochs" \
    --datasets mnist_resnet_20 \
    --arms baseline ${arm} \
    --seeds 33 --epochs 5 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# ═══════════════════════════════════════════════════════════════════
# Image-sibling MLPs (33×15 anchor config)
# ═══════════════════════════════════════════════════════════════════
for arm in scalar_entropy_normalized entropy_target; do
  RUN "FashionMNIST ${arm} dataflow 33-seed × 15 epochs" \
    --datasets fashion_mnist \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

for arm in scalar_entropy_normalized entropy_target; do
  RUN "KMNIST ${arm} dataflow 33-seed × 15 epochs" \
    --datasets kmnist \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# ═══════════════════════════════════════════════════════════════════
# SVHN (15×20 — heavier per-seed)
# ═══════════════════════════════════════════════════════════════════
for arm in scalar_entropy_normalized entropy_target; do
  RUN "SVHN ${arm} dataflow 15-seed × 20 epochs" \
    --datasets svhn \
    --arms baseline ${arm} \
    --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# ═══════════════════════════════════════════════════════════════════
# EMNIST Letters (heaviest — 60k+ samples × 33×15)
# ═══════════════════════════════════════════════════════════════════
for arm in scalar_entropy_normalized entropy_target; do
  RUN "EMNIST Letters ${arm} dataflow 33-seed × 15 epochs" \
    --datasets emnist_letters \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# ═══════════════════════════════════════════════════════════════════
# HighwayMLP-20 (heaviest deep arch — 33×15)
# ═══════════════════════════════════════════════════════════════════
for arm in scalar_entropy_normalized entropy_target; do
  RUN "MNIST HighwayMLP-20 ${arm} dataflow 33-seed × 15 epochs" \
    --datasets mnist_highway_20 \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH7c suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
