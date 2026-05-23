#!/usr/bin/env bash
# Phase-4 — New datasets + Highway-network arch sweep.
#
# Adds to the overnight coverage:
#   A. Tabular sklearn classics: iris, wine, breast_cancer, digits
#      (4 – 64 input dim, 2 – 10 classes). Cheap — 100 seeds × 100 epochs
#      on each is seconds of GPU time. Scalar-entropy AND KL-trajectory,
#      both views.
#   B. HighwayMLP depth sweep on MNIST and FashionMNIST:
#      10- and 20-block Highway networks. Publication power: 33 seeds
#      × 15 epochs, matching the plain-MLP anchor config.

set -u
LOG=/tmp/thesis_iv_views_ph4.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH4 suite started: $(date)" | tee -a $LOG
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
# A  Tabular sklearn datasets  (fast, high-power configs)
# ═══════════════════════════════════════════════════════════════════
for ds in iris wine breast_cancer digits; do
  for view in dataflow factor; do
    RUN "${ds} scalar_entropy ${view} 100-seed × 100 epochs" \
      --datasets ${ds} \
      --arms baseline scalar_entropy \
      --seeds 100 --epochs 100 --lam 0.1 --reg-every-n 10 --view ${view}
    RUN "${ds} kl_trajectory ${view} 100-seed × 100 epochs" \
      --datasets ${ds} \
      --arms baseline kl_trajectory \
      --seeds 100 --epochs 100 --lam 10.0 --reg-every-n 50 --view ${view}
  done
done

# ═══════════════════════════════════════════════════════════════════
# B  HighwayMLP depth sweep — MNIST + FashionMNIST, anchored to the
#    plain-MLP +0.149pp config (33 seeds × 15 epochs, scalar_entropy
#    dataflow).
# ═══════════════════════════════════════════════════════════════════
RUN "MNIST HighwayMLP-10 scalar_entropy dataflow 33-seed × 15 epochs" \
  --datasets mnist_highway_10 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "MNIST HighwayMLP-20 scalar_entropy dataflow 33-seed × 15 epochs" \
  --datasets mnist_highway_20 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "FashionMNIST HighwayMLP-20 scalar_entropy dataflow 33-seed × 15 epochs" \
  --datasets fashion_mnist_highway_20 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

# FashionMNIST ResMLP-20 check — does the MNIST ResMLP null
# still hold on a sibling dataset?
RUN "FashionMNIST ResMLP-20 scalar_entropy dataflow 33-seed × 15 epochs" \
  --datasets fashion_mnist_resnet_20 \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH4 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
