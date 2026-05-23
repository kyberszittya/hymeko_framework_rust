#!/usr/bin/env bash
# Stage D-3-BREAK-VALIDATE — 5-seed validation of C8 winner.
#
# C8 single-seed result (2026-05-21 evening): mAP_50 = 0.0567
# Recipe: lam_gate_neg=2.0, epochs=60, n_box_queries=6,
#         ResNet18-ImageNet, nodelet head, bce gate loss.
# vs published D-3-bis 0.0153 → 3.7× lift on one seed.
#
# This script repeats the recipe at 5 seeds to confirm the win is
# not seed luck.  Falsifier: 5-seed mean below 0.030 (>2× drop)
# OR σ across seeds > 0.020 (>35% relative noise).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_c8_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
JSONL="${OUT_DIR}/5seed.jsonl"
: > "$JSONL"

echo "=== Stage D-3-BREAK-VALIDATE — C8 5-seed ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight VOC / Gömb training.
echo "[orch] waiting for GPU (other VOC / signedkan runs)..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke' | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_seed () {
  local seed="$1"
  local logf="${OUT_DIR}/seed${seed}.log"
  local jsonl_one="${OUT_DIR}/seed${seed}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] SEED ${seed} START" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_c8_5s_${STAMP}_s${seed}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs 60 --input-size 224 \
      --batch-size 8 --n-box-queries 6 \
      --lr 0.003 --seed "$seed" \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind nodelet \
      --lam-gate-neg 2.0 \
      --gate-loss-kind bce \
      --jsonl-out "$jsonl_one" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))

  if [ -s "$jsonl_one" ]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl_one")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
out = pathlib.Path("$JSONL")
with out.open("a") as f:
    for r in rows:
        r["_seed"] = $seed
        f.write(json.dumps(r) + "\n")
PY
  fi
  local m
  m=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$jsonl_one').read_text().splitlines() if l.strip()]
    last = rows[-1] if rows else {}
    m = last.get('mAP_50')
    print(f'{m:.4f}' if isinstance(m, float) else str(m))
except Exception as exc:
    print(f'err:{exc}')
" 2>&1)
  echo "[$(date -Is)] SEED ${seed} DONE rc=$rc mAP_50=$m elapsed=${elapsed}s" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
  run_seed "$SEED"
done

python - <<PY | tee -a "$LOG"
import json, pathlib, statistics
rows = [json.loads(l) for l in pathlib.Path("$JSONL").read_text().splitlines() if l.strip()]
maps = [r["mAP_50"] for r in rows if isinstance(r.get("mAP_50"), float)]
print()
print(f"=== C8 5-seed validation ===")
print(f"  n = {len(maps)}")
if maps:
    mean = statistics.mean(maps)
    sd = statistics.pstdev(maps) if len(maps) >= 2 else 0.0
    print(f"  mAP_50 mean = {mean:.4f}  pstdev = {sd:.4f}")
    print(f"  per-seed: {[f'{m:.4f}' for m in maps]}")
    print()
    # Falsifier checks
    pub_baseline = 0.0153  # published D-3-bis
    if mean < 0.030:
        print(f"  FALSIFIED: mean {mean:.4f} < 0.030 (seed luck verdict)")
    elif sd > 0.020:
        print(f"  HIGH VARIANCE: sd {sd:.4f} > 0.020 — recipe is brittle")
    else:
        lift = mean - pub_baseline
        # paired-z if we had a baseline. Here we report against the
        # published single-seed number; treat lift > 3×sd as a strong claim.
        print(f"  CONFIRMED: mean +{lift:.4f} above published D-3-bis ({pub_baseline:.4f})")
        print(f"  Lift / sd ratio: {lift / max(1e-6, sd):.2f}")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-3-BREAK-VALIDATE DONE $(date -Is) ===" | tee -a "$LOG"
echo "  jsonl: $JSONL" | tee -a "$LOG"
