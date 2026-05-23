#!/usr/bin/env bash
# Phase-17 — Path I λ/L rescaling + more iterations + deeper nets.
#
# ph12 surfaced that Path I's regulariser magnitude scales with L (one
# H_2 term per layer summed in TC). ph14 ran deep archs at λ=0.01,
# which is tuned for L≈4 plain MLPs — for L=20+ that's an L× over-push.
# This phase rescales λ → λ/L and triples the epoch budget so we
# detect any small effect that needs longer iterations to develop.
#
# Combined with the user's hypothesis ("not enough iterations, not deep
# enough"), the test matrix:
#
#   mnist_highway_10 (L≈11, λ_eff = 0.001, 30 epochs)
#     ← control: same arch as ph14, right λ, more iterations
#   mnist_highway_20 (L≈21, λ_eff = 0.0005, 30 epochs)
#     ← ph14's only directional positive (+0.057 ns); rescaled + longer
#   mnist_resnet_40 (L≈81, λ_eff = 0.0001, 30 epochs)
#     ← deepest available (no transformer in this codebase; 40 ResMLP
#       blocks ≈ 81 hookable Linear layers)
#
# 15 seeds (scouting power); the +0.10 acceptance threshold from ph14
# carries over. Acceptance: any combo at Δ ≥ +0.10 → 33-seed power
# follow-up. ~2h on RTX 2070S.

set -u
LOG=/tmp/thesis_iv_views_ph17.log
mkdir -p data/benchmarks

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH17 (Path I λ/L rescale + deeper + more epochs): $(date)" | tee -a $LOG
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

# ─── Control: HighwayMLP-10 at λ/L
RUN "mnist_highway_10 TC mix λ/L=0.001 15×30" \
  --datasets mnist_highway_10 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 30 --lam 0.001 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ─── ph14's directional positive, rescaled + longer
RUN "mnist_highway_20 TC mix λ/L=0.0005 15×30" \
  --datasets mnist_highway_20 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 30 --lam 0.0005 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ─── Deepest available
RUN "mnist_resnet_40 TC mix λ/L=0.0001 15×30" \
  --datasets mnist_resnet_40 --arms baseline total_correlation_mi \
  --seeds 15 --epochs 30 --lam 0.0001 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH17 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
