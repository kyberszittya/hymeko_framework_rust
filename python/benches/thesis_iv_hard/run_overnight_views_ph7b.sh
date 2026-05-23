#!/usr/bin/env bash
# Phase-7b — KA-derived H* and Telgarsky depth-weighted entropy.
#
# Tests two theory-driven new arms on the same 4-dataset universality
# matrix used in phase 6/7:
#   spirals + circles (100×50 each)  — synthetic stress
#   mnist_small (33×15)              — anchor positive
#   mnist_capsnet (33×10)            — anchor negative
#
# Both arms use --target-entropy 0.5 as the safety floor (entropy_target_ka
# auto-overrides this from the KA bound when it's higher). Estimated ~4.5h.

set -u
LOG=/tmp/thesis_iv_views_ph7b.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH7b suite started: $(date)" | tee -a $LOG
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

# Synthetic first (fast)
for arm in entropy_target_ka entropy_telgarsky; do
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

# MNIST plain MLP
for arm in entropy_target_ka entropy_telgarsky; do
  RUN "mnist_small ${arm} dataflow 33-seed × 15 epochs" \
    --datasets mnist_small \
    --arms baseline ${arm} \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

# CapsMLP
for arm in entropy_target_ka entropy_telgarsky; do
  RUN "mnist_capsnet ${arm} dataflow 33-seed × 10 epochs" \
    --datasets mnist_capsnet \
    --arms baseline ${arm} \
    --seeds 33 --epochs 10 --lam 0.1 --reg-every-n 10 --view dataflow \
    --target-entropy 0.5
done

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH7b suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
