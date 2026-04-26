#!/usr/bin/env bash
# Overnight views suite — extension: 4 new datasets.
#
# Intended to run AFTER run_overnight_views.sh completes (GPU time
# does not overlap). Adds four datasets to probe whether the
# plain-MLP spectral-entropy positive from MNIST generalises:
#
#   IMAGE (drop-in MNIST shape, reuses MNISTNetSmall):
#     - FashionMNIST  — visually different 784-dim sibling
#     - KMNIST        — Japanese hiragana, higher stroke density
#
#   SYNTHETIC (2D, reuses SyntheticMLP — 2→32→16→2):
#     - Two Moons     — nonlinear 2-class, classical
#     - Spirals       — hard-for-MLP interlocking spirals
#
# Config rationale:
#   - Image runs match the proven MNIST config: 33 seeds × 15 epochs,
#     λ=0.1, reg-every-10, dataflow view. Directly comparable to the
#     MNIST +0.149% result.
#   - Synthetic runs exploit their speed: 100 seeds × 50 epochs for
#     tight p-values, both views (synthetic runs are cheap).
#
# Estimated budget: ~5–6 hours.

set -u
LOG=/tmp/thesis_iv_views_ext.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views EXT suite started: $(date)" | tee -a $LOG
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
# EXT-1  MNIST-sibling image datasets, plain-MLP config from MNIST
# ═══════════════════════════════════════════════════════════════════

RUN "FashionMNIST plain-MLP dataflow 33-seed × 15 epochs" \
  --datasets fashion_mnist \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "KMNIST plain-MLP dataflow 33-seed × 15 epochs" \
  --datasets kmnist \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

# ═══════════════════════════════════════════════════════════════════
# EXT-2  Synthetic 2D datasets, high-power sweep
# ═══════════════════════════════════════════════════════════════════
# Cheap: ~0.1s/seed/epoch → 100 seeds × 50 epochs ≈ 500 s per run.
# Both views since cost allows.

RUN "Two Moons dataflow 100-seed × 50 epochs" \
  --datasets two_moons \
  --arms baseline scalar_entropy \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "Two Moons factor 100-seed × 50 epochs" \
  --datasets two_moons \
  --arms baseline scalar_entropy \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view factor

RUN "Spirals dataflow 100-seed × 50 epochs" \
  --datasets spirals \
  --arms baseline scalar_entropy \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "Spirals factor 100-seed × 50 epochs" \
  --datasets spirals \
  --arms baseline scalar_entropy \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view factor

# ═══════════════════════════════════════════════════════════════════
# EXT-3  Time-padding additions — close the night with more coverage
# ═══════════════════════════════════════════════════════════════════

# Concentric circles — topologically distinct from moons/spirals
# (closed decision boundary). Both views at high seed count.
RUN "Circles dataflow 100-seed × 50 epochs" \
  --datasets circles \
  --arms baseline scalar_entropy \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "Circles factor 100-seed × 50 epochs" \
  --datasets circles \
  --arms baseline scalar_entropy \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view factor

# FashionMNIST factor view — completes the view comparison on an
# image sibling dataset; matched to MNIST plain-MLP +0.135% factor baseline.
RUN "FashionMNIST plain-MLP factor 33-seed × 15 epochs" \
  --datasets fashion_mnist \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view factor

# ═══════════════════════════════════════════════════════════════════
# EXT-4  Further padding — targeting full-day runtime until ~21:00
# ═══════════════════════════════════════════════════════════════════

# KMNIST factor view — completes view comparison on the 2nd image sibling.
RUN "KMNIST plain-MLP factor 33-seed × 15 epochs" \
  --datasets kmnist \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view factor

# KL-trajectory arm on synthetics — does the alternative regularizer
# beat scalar_entropy on low-dim nonlinear tasks?
RUN "Two Moons KL-trajectory dataflow 100-seed × 50 epochs" \
  --datasets two_moons \
  --arms baseline kl_trajectory \
  --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view dataflow

RUN "Spirals KL-trajectory dataflow 100-seed × 50 epochs" \
  --datasets spirals \
  --arms baseline kl_trajectory \
  --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view dataflow

# CIFAR-10 with longer training — tests the "undertrained baseline"
# hypothesis left open by the previous 20-epoch CIFAR nulls.
RUN "CIFAR-10 plain MLP dataflow 15-seed × 30 epochs" \
  --datasets cifar10 \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 30 --lam 0.1 --reg-every-n 10 --view dataflow

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views EXT suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
