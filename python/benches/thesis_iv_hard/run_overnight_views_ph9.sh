#!/usr/bin/env bash
# Phase-9 — composability vs Batch-Norm, Dropout, Weight-Decay.
#
# For each of 4 stress datasets (spirals, circles, mnist_small,
# mnist_capsnet), run 5 paired comparisons of
#   baseline vs scalar_entropy_normalized
# under 5 different model + optimiser settings:
#
#   V1 vanilla       — base model, Adam(lr=1e-3)
#   V2 dropout       — Dropout(0.5) after each ReLU
#   V3 batch_norm    — BatchNorm1d after each Linear
#   V4 weight_decay  — vanilla model + Adam(weight_decay=1e-4)
#   V5 full_stack    — BN + Dropout + weight_decay (the standard combo)
#
# This tells us whether the entropy regulariser COMPOSES with the
# established techniques — the killer finding for the paper.
#
# Auto-fires after Views PH8 suite finishes.
# Estimated wall-clock: ~11 hours.

set -u
LOG=/tmp/thesis_iv_views_ph9.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH9 suite started: $(date)" | tee -a $LOG
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
# Block A — synthetic 2-D (spirals + circles, 100×50, fast)
# ═══════════════════════════════════════════════════════════════════

# V1 vanilla  ─ baseline vs scalar_entropy_normalized
RUN "spirals V1 vanilla 100×50" \
  --datasets spirals \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "circles V1 vanilla 100×50" \
  --datasets circles \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

# V2 dropout
RUN "spirals V2 dropout 100×50" \
  --datasets spirals_drop \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "circles V2 dropout 100×50" \
  --datasets circles_drop \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

# V3 batch norm
RUN "spirals V3 BN 100×50" \
  --datasets spirals_bn \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "circles V3 BN 100×50" \
  --datasets circles_bn \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

# V4 weight decay (vanilla + Adam(weight_decay=1e-4))
RUN "spirals V4 WD 100×50" \
  --datasets spirals \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

RUN "circles V4 WD 100×50" \
  --datasets circles \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

# V5 full stack (BN + dropout + WD)
RUN "spirals V5 full stack 100×50" \
  --datasets spirals_full \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

RUN "circles V5 full stack 100×50" \
  --datasets circles_full \
  --arms baseline scalar_entropy_normalized \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

# ═══════════════════════════════════════════════════════════════════
# Block B — MNIST plain MLP (33×15, ~75 min/cell)
# ═══════════════════════════════════════════════════════════════════

RUN "mnist_small V1 vanilla 33×15" \
  --datasets mnist_small \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "mnist_small V2 dropout 33×15" \
  --datasets mnist_small_drop \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "mnist_small V3 BN 33×15" \
  --datasets mnist_small_bn \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "mnist_small V4 WD 33×15" \
  --datasets mnist_small \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

RUN "mnist_small V5 full stack 33×15" \
  --datasets mnist_small_full \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

# ═══════════════════════════════════════════════════════════════════
# Block C — CapsMLP MNIST (33×10, ~50 min/cell)
# ═══════════════════════════════════════════════════════════════════

RUN "mnist_capsnet V1 vanilla 33×10" \
  --datasets mnist_capsnet \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "mnist_capsnet V2 dropout 33×10" \
  --datasets mnist_capsnet_drop \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "mnist_capsnet V3 BN 33×10" \
  --datasets mnist_capsnet_bn \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "mnist_capsnet V4 WD 33×10" \
  --datasets mnist_capsnet \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

RUN "mnist_capsnet V5 full stack 33×10" \
  --datasets mnist_capsnet_full \
  --arms baseline scalar_entropy_normalized \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow \
  --weight-decay 1e-4

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH9 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
