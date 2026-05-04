#!/usr/bin/env bash
# Phase-18 — Multi-term wrapping (Lyapunov KL feedback) across deep
# architectures × multiple datasets, with two paths:
#
#   (A) Hand-written PyTorch implementations of the deep architectures
#       on additional datasets (FashionMNIST, KMNIST), to test whether
#       the multi-term wrapping (entropy_lyapunov, λ_a = λ_b = 1)
#       extends the universality picture beyond MNIST.
#
#   (B) The same architectures emitted from HyMeKo descriptions via
#       the torch_dataflow backend (data/nn/mnist_highway_*.hymeko,
#       mnist_resmlp_*.hymeko), to demonstrate that the canonical IR
#       round-trips through to a fully bench-compatible nn.Module
#       and produces qualitatively-comparable regularisation results
#       to the hand-written nets.
#
# entropy_lyapunov implements the Lyapunov-derived schedule
#   λ_eff = λ_0 · exp(-η · KL_step)
#   reg = λ_eff · (λ_a · (H_norm − H*)² + λ_b · H_norm)
# from §5 of the regulariser paper, with the per-term (λ_a, λ_b)
# decomposition.
#
# Estimated runtime: ~5–7h on a 2070 SUPER. Hold for manual
# launch / chain after ph17 finishes.

set -u
LOG=/tmp/thesis_iv_views_ph18.log
mkdir -p data/benchmarks

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH18 (multi-term × deep × HyMeKo descriptions): $(date)" | tee -a $LOG
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
# (A) PyTorch deep archs × {fashion_mnist, kmnist} × multi-term wrap
# ──────────────────────────────────────────────────────────────────────
# Each RUN pairs baseline + entropy_lyapunov (λ_a = λ_b = 1, η = 5).
# 15 seeds × 10 epochs is scouting power — enough to detect a +0.10pp
# effect at the spectral-arm scale. Successful combos earn a 33-seed
# follow-up.

# HighwayMLP-10
RUN "fashion_mnist_highway_10 multi-term 15×10" \
  --datasets fashion_mnist_highway_10 \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

# HighwayMLP-20
RUN "fashion_mnist_highway_20 multi-term 15×10" \
  --datasets fashion_mnist_highway_20 \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

# ResMLP-20
RUN "fashion_mnist_resnet_20 multi-term 15×10" \
  --datasets fashion_mnist_resnet_20 \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

# ──────────────────────────────────────────────────────────────────────
# (B) HyMeKo-generated networks via torch_dataflow backend
# ──────────────────────────────────────────────────────────────────────
# Each network is emitted from data/nn/mnist_<name>.hymeko via
#   target/release/hymeko compile <file> --format torch_dataflow ...
# and wired into run_benchmark.py's DATASETS dictionary as a
# `*_hymeko` entry (see the bottom of run_benchmark.py).
#
# Same multi-term arm (entropy_lyapunov), 15×10 scouting power.

RUN "mnist_highway_3_hymeko multi-term 15×10" \
  --datasets mnist_highway_3_hymeko \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

RUN "mnist_highway_10_hymeko multi-term 15×10" \
  --datasets mnist_highway_10_hymeko \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

RUN "mnist_highway_20_hymeko multi-term 15×10" \
  --datasets mnist_highway_20_hymeko \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

RUN "mnist_resmlp_10_hymeko multi-term 15×10" \
  --datasets mnist_resmlp_10_hymeko \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

RUN "mnist_resmlp_20_hymeko multi-term 15×10" \
  --datasets mnist_resmlp_20_hymeko \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

RUN "mnist_resmlp_40_hymeko multi-term 15×10" \
  --datasets mnist_resmlp_40_hymeko \
  --arms baseline entropy_lyapunov \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH18 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
