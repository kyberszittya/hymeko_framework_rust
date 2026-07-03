#!/usr/bin/env bash
# Chained launch: wait for the running vision compound sweep
# (bdisvucz3 = per_channel + h=64) to free the GPU, then run the
# depth + narrow-breadth HSiKAN sweep. 2026-05-29 user request:
# "schedule testing for depth and otherwise narrow-breadth HSIKANs".
#
# Sweep:
#   hidden ∈ {8, 16}  (narrow-breadth; default = 32)
#   n_layers ∈ {2, 4, 8}  (depth axis; default = 2)
#   = 6 (h, n_layers) configs × 2 datasets × 3 seeds = 36 cells.
#
# Compares to existing baselines at /tmp/vision_strong/ (h=32, L=2)
# and /tmp/vision_h64/ (h=64, L=2) without re-running them. Uses the
# same recipe (full data, 20 epochs, --compile, expandable_segments).
#
# Each orchestrator invocation appends to /tmp/vision_depth_narrow/
# results.jsonl. The chain finally writes a per-(hidden, n_layers,
# dataset) aggregate to summary.json.
set -u
REPO=/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
PY=/home/kyberszittya/miniconda3/bin/python
GATE=/tmp/vision_per_channel_h64/summary.json
OUT=/tmp/vision_depth_narrow
RES="$OUT/results.jsonl"
mkdir -p "$OUT"
cd "$REPO" || exit 3
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[chain $(date -Is)] waiting for bdisvucz3 gate: $GATE"
WAITED=0
while [ ! -f "$GATE" ]; do
  sleep 60; WAITED=$((WAITED+60))
  if [ $WAITED -ge 14400 ]; then
    echo "[chain] gate timeout (4h); aborting to avoid GPU collision"
    exit 4
  fi
done
echo "[chain $(date -Is)] gate cleared after ${WAITED}s; GPU should be free"
sleep 10

# Stage: 6 (h, n_layers) configs × orchestrator.
for h in 8 16; do
  for nL in 2 4 8; do
    LABEL="h${h}_L${nL}"
    echo "[chain $(date -Is)] CONFIG $LABEL"
    systemd-run --user --scope -p MemoryMax=16G \
      $PY -m hymeko_neuro.experiments.runs.run_vision_hypergraph_vs_cnn \
        --models hsikan --datasets mnist,fashion --seeds 0,1,2 \
        --n-epochs 20 --train-subset 0 --batch-size 128 \
        --hidden "$h" --n-layers "$nL" --compile \
        --results-file "$RES" --log-dir "$OUT" \
        > "$OUT/run_${LABEL}.out" 2>&1
    echo "[chain $(date -Is)] $LABEL rc=$?"
  done
done

echo "[chain $(date -Is)] all configs done; final aggregate"
$PY - <<'PY' > "$OUT/summary.json" 2>&1
import json, math
from collections import defaultdict
rows = [json.loads(x) for x in open("/tmp/vision_depth_narrow/results.jsonl") if x.strip()]
by = defaultdict(list)
for r in rows:
    if r.get("test_accuracy") is None: continue
    k = (r["dataset"], int(r["hidden"]), int(r.get("n_layers", 2)))
    by[k].append(r["test_accuracy"])
out = {}
for (ds, h, nL), v in by.items():
    n = len(v); m = sum(v)/n
    sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1)) if n > 1 else 0.0
    out[f"{ds}|h={h}|L={nL}"] = {"mean": m, "sd": sd, "n": n}
print(json.dumps(out, indent=2, sort_keys=True))
PY
echo "[chain $(date -Is)] DONE. summary at $OUT/summary.json"
