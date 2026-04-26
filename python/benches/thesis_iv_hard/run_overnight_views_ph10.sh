#!/usr/bin/env bash
# Phase-10 — entropy_lyapunov closed-loop controller stress test.
#
# Tests the unified Lyapunov-derived dynamic regulariser:
#   lam_eff(t) = lam_0 · exp(-eta · KL(p_{t-1} ‖ p_t))
#   L = lam_eff · (lam_a · (H_norm − H*)² + lam_b · H_norm)
#
# Three framework gaps closed simultaneously:
#   (1) per-term lambda (lam_a, lam_b) — multi-objective explicit
#   (2) KL drives lambda — connects two previously-independent regularisers
#   (3) Lyapunov-safe schedule — derived from V̇ < 0 condition
#
# Math: see V̇ = ⟨∇L_task, ∇L_task⟩ + λ ⟨∇L_task, ∇R⟩ argument.
# Auto-fires after Views PH9 suite finishes. ~3h estimated.

set -u
LOG=/tmp/thesis_iv_views_ph10.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH10 suite started: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG

RUN() {
  local name="$1"; shift
  echo "" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] START: $name" | tee -a $LOG
  echo "  cmd: $*" | tee -a $LOG
  python3 python/benches/thesis_iv_hard/run_benchmark.py "$@" >> $LOG 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE:  $name" | tee -a $LOG \
    || echo "[$(date +%H:%M:%S)] FAIL:  $name (exit=$*)" | tee -a $LOG
}

# Synthetic first
RUN "spirals entropy_lyapunov dataflow 100×50" \
  --datasets spirals \
  --arms baseline entropy_lyapunov \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

RUN "circles entropy_lyapunov dataflow 100×50" \
  --datasets circles \
  --arms baseline entropy_lyapunov \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

# MNIST plain MLP
RUN "mnist_small entropy_lyapunov dataflow 33×15" \
  --datasets mnist_small \
  --arms baseline entropy_lyapunov \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

# CapsMLP
RUN "mnist_capsnet entropy_lyapunov dataflow 33×10" \
  --datasets mnist_capsnet \
  --arms baseline entropy_lyapunov \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH10 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
