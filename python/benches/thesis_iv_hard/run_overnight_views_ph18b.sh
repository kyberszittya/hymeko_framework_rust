#!/usr/bin/env bash
# Phase-18b — Re-run the 6 HyMeKo-generated nets that FAILed in ph18
# due to a 2-D input shape mismatch in the wrapper. Fix landed in
# run_benchmark.py: _wrap_hymeko_net now flattens inputs from
# (B, 28, 28) → (B, 784) before forward().
#
# Same arms / hyperparameters as ph18 bucket B.

set -u
LOG=/tmp/thesis_iv_views_ph18b.log
mkdir -p data/benchmarks

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH18b (HyMeKo-net rerun): $(date)" | tee -a $LOG
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

for ds in mnist_highway_3_hymeko mnist_highway_10_hymeko mnist_highway_20_hymeko \
          mnist_resmlp_10_hymeko mnist_resmlp_20_hymeko mnist_resmlp_40_hymeko; do
  RUN "$ds multi-term 15×10" \
    --datasets $ds \
    --arms baseline entropy_lyapunov \
    --seeds 15 --epochs 10 --lam 0.01 --reg-every-n 50 --view dataflow \
    --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 1.0
done

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH18b finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
