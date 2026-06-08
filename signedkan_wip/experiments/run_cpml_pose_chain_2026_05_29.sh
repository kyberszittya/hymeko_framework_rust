#!/usr/bin/env bash
# Chained launch: wait for the running vision compound sweep
# (bdisvucz3 = per_channel + h=64) to free the GPU, then run the
# CPML-pose Shape A comparison.
#
#   1. Poll for /tmp/vision_per_channel_h64/summary.json (the vision
#      orchestrator's end-of-run aggregate marker).
#   2. GPU smoke: cell_cpml_pose at arity=4, hidden=8, n_epochs=5
#      (sanity gate — abort if it fails).
#   3. Run the 3-seed paired comparison: cell_pose vs cell_cpml_pose
#      on pose_k4 + pose_k6 at hidden=16, n_epochs=80.
#   4. Final aggregate is written to /tmp/cpml_pose/summary.json by the
#      orchestrator itself.
#
# Sequential by design (no GPU contention with the vision sweep). All
# results checkpointed; orchestrator is resumable.
set -u
REPO=/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
PY=/home/kyberszittya/miniconda3/bin/python
GATE=/tmp/vision_per_channel_h64/summary.json
OUT=/tmp/cpml_pose_chain
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

# Stage 1: smoke — one CPMLPose cell at small budget.
echo "[chain $(date -Is)] STAGE 1: CPMLPose GPU smoke (k=4, h=8, 5 ep)"
$PY - <<'PY' > "$OUT/smoke.out" 2>&1
import json, sys, time
import torch
from signedkan_wip.experiments.runs.run_final_cell import cell_cpml_pose
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.monotonic()
row = cell_cpml_pose(arity=4, hidden=8, n_epochs=5, device=device)
wall = time.monotonic() - t0
print(json.dumps(row | {"wall_s": round(wall, 1)} if row else {"row": None}, indent=2))
sys.exit(0 if (row is not None and row["mae"] == row["mae"]) else 5)
PY
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  echo "[chain] SMOKE FAILED rc=$SMOKE_RC (see $OUT/smoke.out)"
  exit 5
fi
echo "[chain $(date -Is)] smoke OK"

# Stage 2: paired 3-seed comparison.
echo "[chain $(date -Is)] STAGE 2: 3-seed paired comparison"
systemd-run --user --scope -p MemoryMax=16G \
  $PY -m signedkan_wip.experiments.runs.run_cpml_pose_compare \
    --arities 4,6 --seeds 0,1,2 --n-epochs 80 --hidden 16 \
    --results-file /tmp/cpml_pose/results.jsonl --log-dir /tmp/cpml_pose \
    > "$OUT/compare.out" 2>&1
COMPARE_RC=$?
echo "[chain $(date -Is)] STAGE 2 done rc=$COMPARE_RC"
echo "[chain $(date -Is)] DONE. Summary at /tmp/cpml_pose/summary.json"
