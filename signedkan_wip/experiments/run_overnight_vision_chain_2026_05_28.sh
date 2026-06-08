#!/usr/bin/env bash
# Overnight chain: wait for the decisive regime run (b01mfrajm) to free the
# GPU, then run the hypergraph-vision vs CNN re-benchmark.
#
#   1. poll until /tmp/regime_abc_slash_k32/summary.json exists (b01mfrajm done)
#   2. GPU smoke (cnn + ricci_stim, tiny) — abort if it fails/OOMs
#   3. fast-four matrix  : cnn,mlp,hgnn,hsikan @ subset 8000, 15 ep, seeds 0-2
#   4. RicciStim matrix  : reduced budget (subset 2000, 10 ep, seeds 0-2) —
#      per-image operator, NOT iso-budget with the fast four (documented)
#   5. final aggregate written to /tmp/vision_bench/summary.json
#
# All cells checkpoint to one jsonl (resumable). GPU is auto-selected by the
# runner once free. Launch under: systemd-run --user --scope -p MemoryMax=16G
set -u
REPO=/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
PY=/home/kyberszittya/miniconda3/bin/python
RES=/tmp/vision_bench/results.jsonl
LOG=/tmp/vision_bench
GATE=/tmp/regime_abc_slash_k32/summary.json
mkdir -p "$LOG"
cd "$REPO" || exit 3
export PYTHONPATH="$REPO"

echo "[chain $(date -Is)] waiting for b01mfrajm gate: $GATE"
WAITED=0
while [ ! -f "$GATE" ]; do
  sleep 60; WAITED=$((WAITED+60))
  if [ $WAITED -ge 14400 ]; then echo "[chain] gate timeout (4h) — aborting to avoid GPU collision"; exit 4; fi
done
echo "[chain $(date -Is)] gate cleared after ${WAITED}s; GPU should be free"
sleep 10

echo "[chain $(date -Is)] GPU smoke (cnn + ricci_stim, tiny)"
$PY -m signedkan_wip.experiments.runs.run_vision_hypergraph_vs_cnn \
    --models cnn,ricci_stim --datasets mnist --seeds 0 \
    --n-epochs 2 --train-subset 256 --batch-size 128 \
    --results-file "$LOG/smoke.jsonl" --log-dir "$LOG/smoke" > "$LOG/smoke.out" 2>&1
if [ $? -ne 0 ]; then echo "[chain] SMOKE FAILED — aborting (see $LOG/smoke.out)"; exit 5; fi
echo "[chain $(date -Is)] smoke OK"

echo "[chain $(date -Is)] fast-four matrix"
$PY -m signedkan_wip.experiments.runs.run_vision_hypergraph_vs_cnn \
    --models cnn,mlp,hgnn,hsikan --datasets mnist,fashion --seeds 0,1,2 \
    --n-epochs 15 --train-subset 8000 --batch-size 128 --hidden 32 \
    --results-file "$RES" --log-dir "$LOG" > "$LOG/fast4.out" 2>&1
echo "[chain $(date -Is)] fast-four rc=$?"

echo "[chain $(date -Is)] RicciStim matrix (reduced budget)"
$PY -m signedkan_wip.experiments.runs.run_vision_hypergraph_vs_cnn \
    --models ricci_stim --datasets mnist,fashion --seeds 0,1,2 \
    --n-epochs 10 --train-subset 2000 --batch-size 128 --hidden 32 \
    --results-file "$RES" --log-dir "$LOG" > "$LOG/ricci.out" 2>&1
echo "[chain $(date -Is)] ricci rc=$?"

echo "[chain $(date -Is)] DONE; final summary at $LOG/summary.json"
$PY -m signedkan_wip.experiments.runs.run_vision_hypergraph_vs_cnn \
    --analyze-only --results-file "$RES" --log-dir "$LOG" | tee "$LOG/summary.json"
