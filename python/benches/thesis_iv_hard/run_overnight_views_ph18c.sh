#!/usr/bin/env bash
# Phase-18c — confirmation sweep for the highway-10 positive +
# λ-grid for depth-fragile variants.
#
# Two questions:
#   (1) Does fashion_mnist_highway_10's +0.087pp positive
#       (ph18 headline, 3-seed) survive a 10-seed sweep?
#   (2) Is the depth-fragility of highway-20 / resmlp-40 real, or
#       was constant-λ off-axis at depth?  Path I's calibration law
#       (λ_multi ~ λ_scalar / L) predicts the original sweep was
#       systematically over-strong.  Test λ ∈ {0.5, 1.0, 2.0, 5.0}
#       at the deeper architectures.
#
# Estimated runtime: 6-8h on a 2070 SUPER.
#
# Output:
#   data/benchmarks/thesis_iv_hard_*.csv  (run_benchmark.py default)
#   /tmp/thesis_iv_views_ph18c.log
set -u
LOG=/tmp/thesis_iv_views_ph18c.log
mkdir -p data/benchmarks

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH18c (highway-10 confirm + depth λ-grid): $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG

RUN() {
  local name="$1"; shift
  echo "" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] START: $name" | tee -a $LOG
  echo "  cmd: $*" | tee -a $LOG
  python3 python/benches/thesis_iv_hard/run_benchmark.py "$@" >> $LOG 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE:  $name" | tee -a $LOG \
    || echo "[$(date +%H:%M:%S)] FAIL:  $name" | tee -a $LOG
}

# ──────────────────────────────────────────────────────────────────────
# (1) highway-10 confirmation: 10-seed sweep at the original λ=1.0
# ──────────────────────────────────────────────────────────────────────
RUN "fashion_mnist_highway_10 confirm 10-seed" \
  --datasets fashion_mnist_highway_10 \
  --arms baseline entropy_lyapunov \
  --seeds 10 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

# ──────────────────────────────────────────────────────────────────────
# (2) λ-grid at depth — highway-20 + resmlp-40 (the null/negative
#     variants from ph18).  Test whether λ ∈ {0.5, 2.0, 5.0} rescues
#     them; if any of those columns turn positive, the depth-fragile
#     framing was a λ-tuning artefact.
# ──────────────────────────────────────────────────────────────────────
for lam_pair in "0.5 0.5" "2.0 2.0" "5.0 5.0"; do
  read lam_a lam_b <<< "$lam_pair"
  RUN "fashion_mnist_highway_20 lam_ab=$lam_a 3-seed" \
    --datasets fashion_mnist_highway_20 \
    --arms baseline entropy_lyapunov \
    --seeds 3 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
    --target-entropy 0.5 --lyapunov-eta 5.0 \
    --lam-a $lam_a --lam-b $lam_b

  RUN "fashion_mnist_resnet_20 lam_ab=$lam_a 3-seed" \
    --datasets fashion_mnist_resnet_20 \
    --arms baseline entropy_lyapunov \
    --seeds 3 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
    --target-entropy 0.5 --lyapunov-eta 5.0 \
    --lam-a $lam_a --lam-b $lam_b

  RUN "mnist_resmlp_40_hymeko lam_ab=$lam_a 3-seed" \
    --datasets mnist_resmlp_40_hymeko \
    --arms baseline entropy_lyapunov \
    --seeds 3 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
    --target-entropy 0.5 --lyapunov-eta 5.0 \
    --lam-a $lam_a --lam-b $lam_b
done

# ──────────────────────────────────────────────────────────────────────
# (3) Cross-check: highway-20 also gets a 10-seed at λ=1.0 to rule
#     out "ph18 had unlucky seeds" before invoking the λ-grid story.
# ──────────────────────────────────────────────────────────────────────
RUN "fashion_mnist_highway_20 10-seed at lam=1.0 cross-check" \
  --datasets fashion_mnist_highway_20 \
  --arms baseline entropy_lyapunov \
  --seeds 10 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH18c done: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
