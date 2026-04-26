#!/usr/bin/env bash
# Phase-11 — cross_layer_mi characterisation.
#
# The 2026-04-26 20:02 spirals run (100 seeds, lam=0.1, dataflow) showed
# Δ = +0.0003 pp, t = +0.16 — a null. Spectral-side arms on the same
# spirals fixture land Δ = +0.4–+0.7 pp (***p<0.001), so activation-side
# Sanchez-Giraldo Renyi-2 mutual information is dramatically weaker than
# spectral entropy at this lambda. Two open questions:
#
#   (Q1) Is the null λ-specific, or does cross_layer_mi truly fail to
#        regularise spirals at every lambda?
#   (Q2) Does cross_layer_mi help on datasets where spectral entropy
#        was inconclusive (kmnist Δ = -0.029, fashion_mnist Δ = +0.039)?
#        Activation-MI may have orthogonal dataset dependence.
#
# Q1 → λ sweep on spirals: λ ∈ {0.01, 0.03, 0.1, 0.3, 1.0}.
#       Baseline runs once at λ=0.1 (deterministic given seed → re-used
#       as the comparison anchor for all other λ values).
# Q2 → cross-dataset paired runs: circles, mnist_small, fashion_mnist,
#       kmnist. Synthetic at 100×50, MNIST family at 33×15. Match the
#       existing suite's protocol so results drop into the same table.
#
# ~2h estimated on the 2070 SUPER. Auto-runs from foreground (no sudo).

set -u
LOG=/tmp/thesis_iv_views_ph11.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH11 (cross_layer_mi characterisation): $(date)" | tee -a $LOG
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

# ---------- Q1: λ sweep on spirals (baseline anchored at λ=0.1) -------
RUN "spirals cross_layer_mi λ=0.1 (anchor + baseline)" \
  --datasets spirals \
  --arms baseline cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "spirals cross_layer_mi λ=0.01" \
  --datasets spirals --arms cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.01 --reg-every-n 10 --view dataflow

RUN "spirals cross_layer_mi λ=0.03" \
  --datasets spirals --arms cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.03 --reg-every-n 10 --view dataflow

RUN "spirals cross_layer_mi λ=0.3" \
  --datasets spirals --arms cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.3 --reg-every-n 10 --view dataflow

RUN "spirals cross_layer_mi λ=1.0" \
  --datasets spirals --arms cross_layer_mi \
  --seeds 100 --epochs 50 --lam 1.0 --reg-every-n 10 --view dataflow

# ---------- Q2: cross-dataset paired runs ----------------------------
# Synthetic — 100×50 to match spirals/circles entries in the suite.
RUN "circles cross_layer_mi 100×50" \
  --datasets circles --arms baseline cross_layer_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow

# MNIST family — 33×15 to match the suite's MNIST plain-MLP entries.
RUN "mnist_small cross_layer_mi 33×15" \
  --datasets mnist_small --arms baseline cross_layer_mi \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow

RUN "fashion_mnist cross_layer_mi 33×15" \
  --datasets fashion_mnist --arms baseline cross_layer_mi \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow

RUN "kmnist cross_layer_mi 33×15" \
  --datasets kmnist --arms baseline cross_layer_mi \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH11 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
