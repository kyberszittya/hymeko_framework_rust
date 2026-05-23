#!/usr/bin/env bash
# Phase-5 — More datasets + CapsMLP architecture.
#
# Adds to the overnight coverage:
#   A. Gaussian quantiles (3-class 2-D synthetic)   — all 4 view×arm combos
#   B. EMNIST Letters (26-class MNIST sibling)      — both views, plain MLP
#   C. SVHN (32×32 colour, CIFAR sibling)           — both views, plain MLP
#   D. CapsMLP on MNIST + FashionMNIST              — new architecture
#      class, with routing-by-agreement.  Spectral regulariser hooks
#      into the primary Linear + routing tensor via spectral_weights().
#
# Ordered fast-first so early results land quickly; heavy CapsMLP runs
# tail. Total ~12 h at the current GPU (Ryzen 7 3700X + RTX 2070S).
#
# Launches after run_overnight_views_ph4.sh writes "Views PH4 suite
# finished" to /tmp/thesis_iv_views_ph4.log.

set -u
LOG=/tmp/thesis_iv_views_ph5.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH5 suite started: $(date)" | tee -a $LOG
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
# A  Gaussian quantiles — 3-class 2-D synthetic
# ═══════════════════════════════════════════════════════════════════
for view in dataflow factor; do
  RUN "Gaussian quantiles scalar_entropy ${view} 100-seed × 50 epochs" \
    --datasets gaussian_quants \
    --arms baseline scalar_entropy \
    --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view ${view}
  RUN "Gaussian quantiles kl_trajectory ${view} 100-seed × 50 epochs" \
    --datasets gaussian_quants \
    --arms baseline kl_trajectory \
    --seeds 100 --epochs 50 --lam 10.0 --reg-every-n 100 --view ${view}
done

# ═══════════════════════════════════════════════════════════════════
# B  EMNIST Letters — 26-class MNIST sibling
# ═══════════════════════════════════════════════════════════════════
RUN "EMNIST Letters scalar_entropy dataflow 33-seed × 15 epochs" \
  --datasets emnist_letters \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "EMNIST Letters scalar_entropy factor 33-seed × 15 epochs" \
  --datasets emnist_letters \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view factor

# ═══════════════════════════════════════════════════════════════════
# C  SVHN — 32×32 colour, CIFAR-10 sibling
# ═══════════════════════════════════════════════════════════════════
RUN "SVHN scalar_entropy dataflow 15-seed × 20 epochs" \
  --datasets svhn \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "SVHN scalar_entropy factor 15-seed × 20 epochs" \
  --datasets svhn \
  --arms baseline scalar_entropy \
  --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 --view factor

# ═══════════════════════════════════════════════════════════════════
# D  CapsMLP — dynamic-routing capsule architecture
# ═══════════════════════════════════════════════════════════════════
# Publication-power-adjacent (33 seeds × 10 epochs per-arm to fit the
# overnight budget; CapsMLP is ~8× slower per step than plain-MLP due
# to the 3-iteration routing loop).

RUN "CapsMLP MNIST scalar_entropy dataflow 33-seed × 10 epochs" \
  --datasets mnist_capsnet \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "CapsMLP FashionMNIST scalar_entropy dataflow 33-seed × 10 epochs" \
  --datasets fashion_mnist_capsnet \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "CapsMLP MNIST scalar_entropy factor 33-seed × 10 epochs" \
  --datasets mnist_capsnet \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view factor

RUN "CapsMLP MNIST kl_trajectory dataflow 33-seed × 5 epochs" \
  --datasets mnist_capsnet \
  --arms baseline kl_trajectory \
  --seeds 33 --epochs 5 --lam 10.0 --reg-every-n 50 --view dataflow

RUN "CapsMLP FashionMNIST scalar_entropy factor 33-seed × 10 epochs" \
  --datasets fashion_mnist_capsnet \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view factor

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH5 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
