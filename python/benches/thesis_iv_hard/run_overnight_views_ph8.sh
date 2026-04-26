#!/usr/bin/env bash
# Phase-8 — cross_layer_mi (Sanchez-Giraldo Renyi-2 mutual information)
# stress test on the 4-dataset universality matrix.
#
# Tests whether the activation-side cross-layer MI penalty preserves the
# universality property established by Path B and Path A:
#   - amplifies positives (spirals, MNIST plain MLP)
#   - neutralises negatives (circles, CapsMLP MNIST)
#
# Math: reports/sanchez_giraldo_framework.{md,pdf}
# Sanity: python/benches/thesis_iv_hard/test_sanchez_giraldo.py
#
# Auto-fires after Views PH7c suite finishes.
# Estimated wall-clock: ~3h.

set -u
LOG=/tmp/thesis_iv_views_ph8.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH8 suite started: $(date)" | tee -a $LOG
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

# Synthetic first
RUN "spirals cross_layer_mi dataflow 100-seed × 50 epochs" \
  --datasets spirals \
  --arms baseline cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "circles cross_layer_mi dataflow 100-seed × 50 epochs" \
  --datasets circles \
  --arms baseline cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

# MNIST plain MLP
RUN "mnist_small cross_layer_mi dataflow 33-seed × 15 epochs" \
  --datasets mnist_small \
  --arms baseline cross_layer_mi \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

# CapsMLP
RUN "mnist_capsnet cross_layer_mi dataflow 33-seed × 10 epochs" \
  --datasets mnist_capsnet \
  --arms baseline cross_layer_mi \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH8 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
