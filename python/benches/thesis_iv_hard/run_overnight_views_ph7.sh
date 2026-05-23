#!/usr/bin/env bash
# Phase-7 — H* sweep on Path A + three combined arms on the universality matrix.
#
# Two questions:
#   1. Is H*=0.5 the best target, or is there a better fixed value?
#      → H* sweep on synthetic (cheap) at {0.0, 0.25, 0.5, 0.75, 1.0}.
#   2. Do the integrated arms (A+B blend, adaptive-λ, A+B+C sum)
#      beat the standalone B/A?
#      → run all three on the four phase-6 datasets (dataflow only).
#
# Estimated total: ~7.5h.

set -u
LOG=/tmp/thesis_iv_views_ph7.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH7 suite started: $(date)" | tee -a $LOG
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
# A  H* sweep on Path A — synthetic only (cheap)
# ═══════════════════════════════════════════════════════════════════
for hstar in 0.00 0.25 0.50 0.75 1.00; do
  for ds in spirals circles; do
    RUN "${ds} entropy_target H*=${hstar} 100-seed × 50 epochs" \
      --datasets ${ds} \
      --arms baseline entropy_target \
      --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
      --target-entropy ${hstar}
  done
done

# ═══════════════════════════════════════════════════════════════════
# B  Three combined arms on the four phase-6 stress datasets
# ═══════════════════════════════════════════════════════════════════
# Synthetic first (fast → early signals)
for arm in entropy_unified entropy_adaptive total_combined; do
  RUN "spirals ${arm} dataflow 100-seed × 50 epochs" \
    --datasets spirals \
    --arms baseline ${arm} \
    --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
  RUN "circles ${arm} dataflow 100-seed × 50 epochs" \
    --datasets circles \
    --arms baseline ${arm} \
    --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# Then MNIST (33×15) and CapsMLP (33×10) — heavier
for arm in entropy_unified entropy_adaptive total_combined; do
  RUN "mnist_small ${arm} dataflow 33-seed × 15 epochs" \
    --datasets mnist_small \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

for arm in entropy_unified entropy_adaptive total_combined; do
  RUN "mnist_capsnet ${arm} dataflow 33-seed × 10 epochs" \
    --datasets mnist_capsnet \
    --arms baseline ${arm} \
    --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH7 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
