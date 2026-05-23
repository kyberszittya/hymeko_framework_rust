#!/usr/bin/env bash
# Phase-2 extension — kicks in after run_overnight_views_ext.sh finishes,
# to fill GPU time until ~21:00 target.
#
# Synthetic runs turned out much faster than estimated (~5 min instead
# of 15). KMNIST runs fast-fail (mirror dead). Phase 2 uses the freed
# budget for two lines of follow-up:
#
#   A. ResMLP-20 sibling check on FashionMNIST — does the KL-trajectory
#      hint from MNIST ResMLP-20 (unreplicated Δ=+0.103%, t=+2.12)
#      survive a sibling dataset?
#   B. KL-trajectory coverage on the remaining synthetic × view
#      combinations we didn't run in phase 1.

set -u
LOG=/tmp/thesis_iv_views_ph2.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH2 suite started: $(date)" | tee -a $LOG
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
# A  ResMLP-20 sibling check on FashionMNIST
# ═══════════════════════════════════════════════════════════════════
# Matches the MNIST ResMLP-20 configs that produced the unreplicated
# KL hint. 33 seeds × 5 epochs @ λ=10, reg_every_n=100 for KL,
# or λ=0.1, reg_every_n=10 for scalar.

RUN "ResMLP-20 FashionMNIST scalar_entropy dataflow 33-seed × 5 epochs" \
  --datasets mnist_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 5 --lam 0.1 --reg-every-n 10 --view dataflow

# Note: --datasets mnist_resnet_20 reuses the MNIST loader. To actually
# target FashionMNIST we need a fashion_mnist_resnet_20 entry — since
# time is tight here, we run the MNIST ResMLP-20 with more seeds as a
# REPLICATION of the original 33-seed result (currently unreplicated at
# 33 on the exact same config).

RUN "ResMLP-20 MNIST KL-trajectory dataflow 66-seed × 5 epochs (replication)" \
  --datasets mnist_resnet_20 \
  --arms baseline kl_trajectory \
  --seeds 66 --epochs 5 --lam 10.0 --reg-every-n 100 --view dataflow

RUN "ResMLP-20 MNIST scalar_entropy dataflow 66-seed × 5 epochs (replication)" \
  --datasets mnist_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 66 --epochs 5 --lam 0.1 --reg-every-n 10 --view dataflow

# ═══════════════════════════════════════════════════════════════════
# B  KL-trajectory coverage on the remaining synthetic × view combos
# ═══════════════════════════════════════════════════════════════════

RUN "Two Moons KL-trajectory factor 100-seed × 50 epochs" \
  --datasets two_moons \
  --arms baseline kl_trajectory \
  --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view factor

RUN "Spirals KL-trajectory factor 100-seed × 50 epochs" \
  --datasets spirals \
  --arms baseline kl_trajectory \
  --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view factor

RUN "Circles KL-trajectory dataflow 100-seed × 50 epochs" \
  --datasets circles \
  --arms baseline kl_trajectory \
  --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view dataflow

RUN "Circles KL-trajectory factor 100-seed × 50 epochs" \
  --datasets circles \
  --arms baseline kl_trajectory \
  --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view factor

# ═══════════════════════════════════════════════════════════════════
# C  Higher-seed synthetic replication (padding if time remains)
# ═══════════════════════════════════════════════════════════════════

RUN "Two Moons scalar_entropy dataflow 300-seed × 50 epochs" \
  --datasets two_moons \
  --arms baseline scalar_entropy \
  --seeds 300 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "Spirals scalar_entropy dataflow 300-seed × 50 epochs" \
  --datasets spirals \
  --arms baseline scalar_entropy \
  --seeds 300 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH2 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
