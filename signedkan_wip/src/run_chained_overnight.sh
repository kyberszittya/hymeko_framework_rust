#!/usr/bin/env bash
# Chainer: wait for the camera-ready 5-seed bench to finish, then
# launch ph18c (entropy seed-sweep + λ-grid).
#
# Detection: the camera-ready bench writes
# `signedkan_wip/experiments/results/overnight_camera_ready.jsonl`,
# 75 lines expected on success.  We poll every 2 minutes until either
# (a) the file has >= 75 lines, or (b) no `run_final_cell` Python
# process is alive and the file has stopped growing for >5 minutes.
# Then we wait one extra minute for the GPU to release, and launch
# the ph18c sweep.
#
# Usage:
#   nohup bash signedkan_wip/src/run_chained_overnight.sh \
#         > /tmp/chained_overnight.log 2>&1 &
set -u

CR_FILE="signedkan_wip/experiments/results/overnight_camera_ready.jsonl"
EXPECTED_CELLS=75

echo "[$(date +%H:%M:%S)] chainer started; waiting for camera-ready bench"

# Phase 1 — wait for camera-ready bench
last_size=-1
last_change=$(date +%s)
while true; do
  cur_size=$(wc -l < "$CR_FILE" 2>/dev/null || echo 0)
  if [[ "$cur_size" -ge "$EXPECTED_CELLS" ]]; then
    echo "[$(date +%H:%M:%S)] camera-ready bench complete ($cur_size cells)"
    break
  fi
  # Detect a stall: file unchanged for 5 minutes AND no python writer alive.
  now=$(date +%s)
  if [[ "$cur_size" != "$last_size" ]]; then
    last_size="$cur_size"
    last_change="$now"
  fi
  age=$(( now - last_change ))
  if [[ "$age" -gt 300 ]]; then
    # Use exact arg matching to avoid pgrep matching its own command line.
    if ! pgrep -f "signedkan_wip\.src\.run_final_cell" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] camera-ready bench stalled at $cur_size " \
           "cells (no writer for ${age}s); proceeding anyway"
      break
    fi
  fi
  sleep 120
done

# Phase 2 — wait for GPU release
echo "[$(date +%H:%M:%S)] waiting 60s for GPU to release"
sleep 60

# Phase 3 — launch ph18c
echo "[$(date +%H:%M:%S)] launching ph18c sweep"
bash python/benches/thesis_iv_hard/run_overnight_views_ph18c.sh
echo "[$(date +%H:%M:%S)] ph18c sweep finished"

# Phase 4 — also queue the sinusoid-control baselines if we have time
# (these are in the roadmap but not yet implemented; placeholder for
# tomorrow)
echo "[$(date +%H:%M:%S)] chainer finished; sinusoid-controls deferred"
