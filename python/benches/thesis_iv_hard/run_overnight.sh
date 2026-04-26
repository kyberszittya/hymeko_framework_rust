#!/usr/bin/env bash
# Overnight benchmark suite for thesis IV extensions.
#
# Designed to produce a paper-ready dataset by morning. Experiments are
# ordered so the highest-priority / most-informative ones run first, in
# case the suite gets interrupted.
#
# Total estimated budget: ~7–9 hours on RTX 2070 Super.
#
# Outputs: timestamped CSVs under data/benchmarks/, one per experiment.
# Each run is tagged with a prefix in the filename for grouping.
#
# To follow progress: `tail -f /tmp/thesis_iv_overnight.log`

set -u  # bail on unset vars, but NOT on error — we want to continue if one experiment fails

LOG=/tmp/thesis_iv_overnight.log
OUT=data/benchmarks
mkdir -p $OUT

# Tag each experiment's output with an explicit prefix via --out-dir trick
# (run_benchmark writes to $OUT/thesis_iv_hard_<timestamp>.csv; we just
# rely on ordering to identify them).

echo "==================================" | tee -a $LOG
echo "Thesis IV overnight suite started: $(date)" | tee -a $LOG
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
# PRIORITY 1 — CIFAR-10 generalization (biggest missing paper datum)
# ═══════════════════════════════════════════════════════════════════

# CIFAR-10 plain MLP + ResMLP depth sweep: baseline vs scalar_entropy.
# Expect ~40-60s per run. 4 datasets × 2 arms × 15 seeds = 120 runs.
RUN "CIFAR10 depth sweep @ λ=0.1" \
  --datasets cifar10_small cifar10_resnet_3 cifar10_resnet_10 cifar10_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 10 --lam 0.1 --reg-every-n 10

# CIFAR-10 plain MLP thesis-scale style (3072→16→8→10 won't fit CIFAR well,
# use wider). Still the "simple compact MLP" reference.
RUN "CIFAR10 λ sweep on plain (cifar10)" \
  --datasets cifar10 \
  --arms baseline scalar_entropy l2_weight_decay \
  --seeds 15 --epochs 10 --lam 0.1 --reg-every-n 10

# ═══════════════════════════════════════════════════════════════════
# PRIORITY 2 — Resolve 40-block inverted-U vs underfitting
# ═══════════════════════════════════════════════════════════════════

# 40-block at 15 epochs — does the effect come back with more training?
RUN "MNIST ResNet-40 at 15 epochs" \
  --datasets mnist_resnet_40 \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 15 --lam 0.1 --reg-every-n 10

# Also extend the best-performing setups to longer training for confirmation
RUN "MNIST plain + ResMLP-20 at 15 epochs" \
  --datasets mnist_small mnist_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 20 --epochs 15 --lam 0.1 --reg-every-n 10

# ═══════════════════════════════════════════════════════════════════
# PRIORITY 3 — KL-trajectory cement of negative result
# ═══════════════════════════════════════════════════════════════════

# The λ sweep on 2026-04-22 showed KL hurts at every λ (significantly
# negative at λ=1000, t=-4.85). Cement that finding with larger seed
# count at the most-informative λ, then on ResMLP-20 for generality.
# This makes the negative result publication-quality.
RUN "MNIST KL-trajectory 33-seed cement at λ=10 cadence=100" \
  --datasets mnist_small \
  --arms baseline kl_trajectory \
  --seeds 33 --epochs 5 --lam 10.0 --reg-every-n 100

RUN "MNIST ResMLP-20 KL-trajectory λ=10 cadence=100" \
  --datasets mnist_resnet_20 \
  --arms baseline kl_trajectory \
  --seeds 15 --epochs 5 --lam 10.0 --reg-every-n 100

# ═══════════════════════════════════════════════════════════════════
# PRIORITY 4 — Highway depth sweep complement
# ═══════════════════════════════════════════════════════════════════

RUN "MNIST Highway depth sweep" \
  --datasets mnist_highway mnist_highway_10 mnist_highway_20 \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 5 --lam 0.1 --reg-every-n 10

# ═══════════════════════════════════════════════════════════════════
# PRIORITY 5 — 33-seed publication-quality confirmations at key setups
# ═══════════════════════════════════════════════════════════════════

# Final cements for the paper table.
RUN "MNIST plain 33-seed at 15 epochs" \
  --datasets mnist_small \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10

RUN "MNIST ResMLP-20 33-seed at 15 epochs" \
  --datasets mnist_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Overnight suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "" | tee -a $LOG
echo "CSVs in: $OUT/" | tee -a $LOG
ls -la $OUT/thesis_iv_hard_*.csv | tail -20 | tee -a $LOG
