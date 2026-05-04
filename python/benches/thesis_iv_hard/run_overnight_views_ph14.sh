#!/usr/bin/env bash
# Phase-14 — Path I (total_correlation_mi) × deep architectures.
#
# ph12 covered Path I × plain MLP across spirals + circles + 3 MNIST
# siblings. ph13 covers Path I × CapsMLP MNIST (the historical
# significant-negative). This phase extends to the *deep* architectures
# in the universality programme: ResMLP-20 and HighwayMLP variants,
# across MNIST and FashionMNIST.
#
# Hypothesis: Path I bites *harder* on deep nets than Path A/B because
# multi-way redundancy across L=20 layers is what TC (total correlation)
# is actually designed to penalise — pairwise spectral entropy can't
# express L-way joint structure.
#
# Compute budget — this is a SCOUTING sweep, not a power study:
#   - 15 seeds (vs 33 in the suite) — first-pass detection of effect
#     size, not p-value chasing.
#   - 5–10 epochs (vs 15 in the suite) — capture training-trajectory
#     direction, not steady-state.
#   - mode = mix (the principled stage-aware variance-momentum default).
#
# If ph14 surfaces a ≥ +0.10 pp Δ at 15 seeds, follow up with a
# full-power 33×15 paired run on that arch×dataset combo.
#
# Estimated runtime ~2.5–3 h on a 2070 SUPER. Hold for manual launch
# until ph12 (and optionally ph13) results have been reviewed — if
# Path I is a bust on plain MLPs, the deep-arch sweep is wasted compute.

set -u
LOG=/tmp/thesis_iv_views_ph14.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH14 (Path I × deep archs): $(date)" | tee -a $LOG
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

# ---------- ResMLP-20 (residual depth, hidden=16) -----------------
# ResMLP-20 has 20 residual blocks → ~21 spectral_weights() Linear
# layers. Path I forms a 21-way joint Gram (renormalised stepwise).
# Per-step cost grows ~linearly with L.

RUN "mnist_resnet_20 TC mix 15×10" \
  --datasets mnist_resnet_20 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "fashion_mnist_resnet_20 TC mix 15×10" \
  --datasets fashion_mnist_resnet_20 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- HighwayMLP-10 (gated, fewer blocks for budget) ---------
# HighwayMLP-10 has 10 highway blocks → ~11 spectral_weights().
# Cheaper than -20 for the first scout; if positive, follow up with -20.

RUN "mnist_highway_10 TC mix 15×15" \
  --datasets mnist_highway_10 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "fashion_mnist_highway_10 TC mix 15×15" \
  --datasets fashion_mnist_highway_10 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- HighwayMLP-20 (full depth, only on MNIST to scout) -----
# Skip on FashionMNIST for budget; reinstated if -10 is positive.

RUN "mnist_highway_20 TC mix 15×10" \
  --datasets mnist_highway_20 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- CapsMLP × FashionMNIST — DROPPED 2026-04-27 ------------
# Reason: ph13 (Path I × CapsMLP MNIST) revealed that CapsMLP's
# spectral_weights() returns a synthesised tensor (W.permute().reshape())
# alongside the primary Linear weight — only the primary Linear has a
# matching `nn.Linear.weight is W` for the activation hook to find.
# Path F / Path I require ≥ 2 hookable modules; CapsMLP has exactly 1,
# so the regulariser silently no-ops and the arm collapses to baseline.
# The `fashion_mnist_capsnet` run would suffer the same fate. Drop until
# CapsMLP is refactored to expose the routing as an nn.Module submodule
# whose forward output can be hooked.

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH14 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
