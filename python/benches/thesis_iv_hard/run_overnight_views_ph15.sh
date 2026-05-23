#!/usr/bin/env bash
# Phase-15 — fine-grained λ sweep on Path I to nail down the productive band.
#
# ph12 surfaced a faint positive trend at λ=0.03 (Δ=+0.306 pp .) and a
# trend-positive at λ=0.01 (Δ=+0.182 pp ns), with collapse from λ=0.1
# upward. ph15 brackets the [0.005, 0.05] band at finer resolution to
# locate the maximum-Δ operating point on spirals — the same fixture
# Path A/B got their +0.4–0.6 pp wins on.
#
# Goal: produce a 5-point λ curve so any future Path I run can use a
# defensible default. Not a power study — n=100 at 50 epochs gives ~±0.1
# pp resolution at this fixture, which is enough to bracket the optimum.
#
# Mode = mix only (damp is degenerate per ph12; amplify under-performs
# mix everywhere). β = 0.9 (default — varying β is ph12-Q4's job).
#
# baseline anchor reused from ph12-Q1 spirals damp run if available;
# otherwise a fresh paired run is shipped.
#
# ~25 min on RTX 2070 SUPER.

set -u
LOG=/tmp/thesis_iv_views_ph15.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH15 (Path I fine-grained λ sweep): $(date)" | tee -a $LOG
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

# Anchor: paired baseline + TC at the centre of the band so ph15 is
# self-contained even if older anchor CSVs are pruned.
RUN "spirals TC mix λ=0.02 (anchor + baseline)" \
  --datasets spirals \
  --arms baseline total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.02 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# Bracketing the productive band, single-arm (deterministic baseline
# pairs with ph15's anchor row by seed).
RUN "spirals TC mix λ=0.005" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.005 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "spirals TC mix λ=0.01" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.01 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "spirals TC mix λ=0.03" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.03 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "spirals TC mix λ=0.05" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.05 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH15 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
