#!/usr/bin/env bash
# Phase-13 — Path I (total_correlation_mi) × CapsMLP MNIST.
#
# CapsMLP MNIST is the historical "significant negative" under
# unnormalised spectral entropy (Δ = −0.057 pp, p < 0.05; phases 1-5).
# Path B (`scalar_entropy_normalized`) neutralised it (Δ ≈ −0.002 ns).
# This phase tests whether Path I reproduces the neutralisation, and
# ideally turns the sign positive — the hypothesis being that capsule
# routing creates non-trivial cross-layer redundancy which TC penalises
# directly.
#
# Three runs, mirror ph12's structure on the diagnostic dataset:
#   PH13-Q1: mode head-to-head (damp vs amplify vs mix), λ=0.1
#   PH13-Q2: λ sweep at mode=mix on the same fixture
#
# CapsMLP has 3 capsule routing iterations + squash; per-seed runtime
# ~1.5–2× plain MLP. Estimated 30–40 min.

set -u
LOG=/tmp/thesis_iv_views_ph13.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH13 (Path I × CapsMLP MNIST): $(date)" | tee -a $LOG
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

# ---------- Q1: mode head-to-head on CapsMLP ----------
# 33×10 matches the ph6 CapsMLP entry in RESULTS_VIEWS_SUITE.md.
RUN "mnist_capsnet TC mode=damp (anchor + baseline)" \
  --datasets mnist_capsnet --arms baseline total_correlation_mi \
  --seeds 33 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode damp

RUN "mnist_capsnet TC mode=amplify" \
  --datasets mnist_capsnet --arms total_correlation_mi \
  --seeds 33 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode amplify

RUN "mnist_capsnet TC mode=mix" \
  --datasets mnist_capsnet --arms total_correlation_mi \
  --seeds 33 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- Q2: λ sweep on CapsMLP at mode=mix --------
RUN "mnist_capsnet TC mix λ=0.001" \
  --datasets mnist_capsnet --arms total_correlation_mi \
  --seeds 33 --epochs 10 --lam 0.001 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "mnist_capsnet TC mix λ=0.1" \
  --datasets mnist_capsnet --arms total_correlation_mi \
  --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH13 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
