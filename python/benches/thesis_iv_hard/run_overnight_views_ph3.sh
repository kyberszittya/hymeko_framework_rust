#!/usr/bin/env bash
# Phase-3 — KMNIST runs with the redesigned HF-parquet-backed loader.
#
# The torchvision KMNIST URL (codh.rois.ac.jp) has been dead since the
# overnight suite started, so all KMNIST slots in phase 1 + 2 fast-failed.
# This phase uses the new `kmnist_loaders` in run_benchmark.py which
# reads from tanganke/kmnist parquet already cached under data/kmnist/.
#
# Launches after run_overnight_views_ph2.sh writes "Views PH2 suite
# finished" to /tmp/thesis_iv_views_ph2.log.

set -u
LOG=/tmp/thesis_iv_views_ph3.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH3 (KMNIST) started: $(date)" | tee -a $LOG
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

# Primary KMNIST sibling check — same config as the MNIST +0.149% finding
# and the FashionMNIST sibling run from phase 1.
RUN "KMNIST plain-MLP dataflow 33-seed × 15 epochs" \
  --datasets kmnist \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view dataflow

RUN "KMNIST plain-MLP factor 33-seed × 15 epochs" \
  --datasets kmnist \
  --arms baseline scalar_entropy \
  --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 --view factor

RUN "KMNIST plain-MLP KL-trajectory dataflow 33-seed × 5 epochs" \
  --datasets kmnist \
  --arms baseline kl_trajectory \
  --seeds 33 --epochs 5 --lam 10.0 --reg-every-n 100 --view dataflow

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "Views PH3 suite finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
