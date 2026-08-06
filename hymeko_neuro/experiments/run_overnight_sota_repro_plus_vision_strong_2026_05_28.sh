#!/usr/bin/env bash
# Overnight chain (2026-05-28 night):
#   1. SOTA reproduction: Slashdot edge_cr kernel-ON 5-seed
#      (the May-09 SOTA 0.9070 +/- .0029) under today's code + torch 2.11.
#      Preserves the May-09 result file; writes today's run to a new dated
#      jsonl; emits a paired comparison vs May-09.
#   2. Stronger hypergraph-vision rebench (autonomous follow-up to the
#      2026-05-28 night-1 run): full data (subset=0), 20 epochs (vs 15 last
#      round), h=32, 4 fast models, 3 seeds. Skips RicciStim (its per-image
#      forward dominates wall and was already shown not to close the gap).
#
# Both stages are sequential -> no GPU contention. Each stage owns its own
# error handling. The vision stage runs even if the SOTA stage fails (they
# are independent questions).
set -u
REPO=/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
PY=/home/kyberszittya/miniconda3/bin/python
cd "$REPO" || exit 3
export PYTHONPATH="$REPO"

OUT=/tmp/overnight_2026_05_28_night2
mkdir -p "$OUT"

echo "[chain $(date -Is)] STAGE 1: SOTA reproduction (Slashdot edge_cr kernel-ON 5-seed)"
bash hymeko_neuro/experiments/run_slashdot_edge_cr_kernel_on_repro_2026_05_28.sh \
    > "$OUT/sota.out" 2>&1
SOTA_RC=$?
echo "[chain $(date -Is)] STAGE 1 done rc=$SOTA_RC"

echo "[chain $(date -Is)] STAGE 2: stronger vision rebench (full data, h=32, 20ep)"
$PY -m hymeko_neuro.experiments.runs.run_vision_hypergraph_vs_cnn \
    --models cnn,mlp,hgnn,hsikan --datasets mnist,fashion --seeds 0,1,2 \
    --n-epochs 20 --train-subset 0 --batch-size 128 --hidden 32 \
    --results-file /tmp/vision_strong/results.jsonl \
    --log-dir /tmp/vision_strong \
    > "$OUT/vstrong.out" 2>&1
echo "[chain $(date -Is)] STAGE 2 done rc=$?"

echo "[chain $(date -Is)] DONE"
