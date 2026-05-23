#!/usr/bin/env bash
# Stage D-3-BREAK Phase 8 — 320 px SGD-step-count probe.
#
# Phase 7 verdict (2026-05-22 evening): lazy DataLoader is clean
# (B7 0.0880 in C9 band) but B8 (320 px, b=8, ep=90) collapsed to
# 0.0257 with cls_acc=0.  Loss-drop halved (41.8% → 24.2%) while
# per-step wall is sub-quadratic in pixel area — consistent with
# *undertraining* at the new target dimension, not architectural
# ceiling.
#
# Phase 8 (overnight): two orthogonal step-count probes at 320 px:
#
#   B9_long_ep180     b=8, ep=180, 320 px — 2× SGD steps via epochs
#   B10_small_batch   b=4, ep=90,  320 px — 2× SGD steps via batch
#
# Single seed each (Phase-style falsifier; promote to 5-seed only if
# either lifts above C9).  Falsifier:
#
#   any cell > 0.0790 (C9 mean) ⇒ step count is the lever and 320 px
#     is productive; promote winner to 5-seed
#   both cells ≤ C9 band         ⇒ 320 px is an architectural
#     ceiling at 714k params; lever exhausted
#   any cell with cls_acc > 0    ⇒ K+1 softmax did not collapse;
#     the optimisation got somewhere
#
# Expected wall ≈ 5000 s + 3750 s = ~2.5 h.  Overnight-safe with
# 16 GiB cgroup MemoryMax.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_320px_stepcount_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 8 — 320 px step-count probe ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight VOC / Gömb training.
echo "[orch] waiting for GPU..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke|ros2 launch hymeko_ros2_demo' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_cell () {
  local cell="$1"; shift
  local input="$1"; shift
  local batch="$1"; shift
  local epochs="$1"; shift

  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START input=${input} batch=${batch} ep=${epochs}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_320px_${STAMP}_${cell}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs "$epochs" --input-size "$input" \
      --batch-size "$batch" --n-box-queries 6 \
      --lr 0.003 --seed 0 \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind nodelet \
      --lam-gate-neg 2.0 \
      --gate-loss-kind bce \
      --jsonl-out "$jsonl" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))

  if [ -s "$jsonl" ]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
out = pathlib.Path("$GRID_JSONL")
with out.open("a") as f:
    for r in rows:
        r["_cell"]   = "$cell"
        r["_input"]  = $input
        r["_batch"]  = $batch
        r["_epochs"] = $epochs
        f.write(json.dumps(r) + "\n")
PY
  fi

  local m
  m=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$jsonl').read_text().splitlines() if l.strip()]
    last = rows[-1] if rows else {}
    m = last.get('mAP_50')
    print(f'{m:.4f}' if isinstance(m, float) else str(m))
except Exception as exc:
    print(f'err:{exc}')
" 2>&1)
  echo "[$(date -Is)] CELL ${cell} DONE rc=$rc mAP_50=$m elapsed=${elapsed}s" | tee -a "$LOG"
}

# Two cells — long-epoch then small-batch at 320 px.
run_cell B9_long_ep180   320  8  180
run_cell B10_small_batch 320  4   90

# Aggregator
python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C9_MEAN = 0.0790
C9_SD   = 0.0105
B8_320  = 0.0257  # Phase 7 reference at b=8, ep=90, 320 px

rows.sort(key=lambda r: -r["mAP_50"])
print()
print(f"=== Phase 8 320 px step-count probe ===")
print(f"{'cell':16s} {'input':>5s} {'batch':>5s} {'epochs':>6s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s}")
for r in rows:
    print(f"{r['_cell']:16s} {r['_input']:>5d} {r['_batch']:>5d} {r['_epochs']:>6d} "
          f"{r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f}")
print()
print(f"C9 5-seed anchor: mean = {C9_MEAN:.4f}  pstdev = {C9_SD:.4f}")
print(f"                  band  [{C9_MEAN-C9_SD:.4f}, {C9_MEAN+C9_SD:.4f}]")
print(f"Phase 7 B8 (320, b=8, ep=90) reference: {B8_320:.4f}")
print()
for r in rows:
    cell = r["_cell"]
    m = r["mAP_50"]
    if m > C9_MEAN + C9_SD:
        print(f"  {cell}: mAP={m:.4f}  ✓ ABOVE C9 band — step count IS the lever; promote to 5-seed")
    elif m > C9_MEAN - C9_SD:
        print(f"  {cell}: mAP={m:.4f}  · in C9 band — step count recovers but no lift")
    elif m > B8_320 * 2:
        print(f"  {cell}: mAP={m:.4f}  · 2× over B8 but below C9 band — partial recovery")
    else:
        print(f"  {cell}: mAP={m:.4f}  ✗ ≤ 2× B8 — 320 px ceiling holds at 714k params")
PY

# Flag OOMs
for cl in B9 B10; do
  log=$(ls $OUT_DIR/${cl}_*.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  if grep -q 'CUDA out of memory' "$log" 2>/dev/null; then
    echo "[orch] $(basename $log .log) hit OOM" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== Phase 8 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
