#!/usr/bin/env bash
# Stage D-3-BREAK Phase 7 — lazy DataLoader refactor validation.
#
# Phase 6 verdict (2026-05-22 morning): backbone-shrink hurts mAP;
# the 8 GiB GPU is capped at 224 px because train_voc_stagec.py
# pre-loaded the entire VOC set as one GPU tensor (6.16 GiB at 320 px).
#
# Phase 7 refactor (this afternoon): X stays on CPU; train_one_config
# and compute_detection_metrics transfer per-batch with .to(device,
# non_blocking=True).  Cluttered MNIST smoke confirms no regression.
#
# Two validation cells:
#   B7  C9 recipe at 224 px       — reproduces ~0.0790 mAP_50 (sanity)
#   B8  C9 recipe at 320 px       — the previously-blocked probe
#
# Single seed each.  Falsifier:
#   B7 mAP_50 in C9 5-seed band [0.0685, 0.0895] = refactor clean
#   B8 mAP_50 > 0.0790 = resolution axis genuinely unblocked
#   B8 OOM = the lazy refactor is incomplete (likely activations
#            now the binding cap)

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_lazy_loader_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 7 — lazy loader validation ===" | tee -a "$LOG"
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
    --unit="voc_lazy_${STAMP}_${cell}.scope" \
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

# Two cells — sanity at 224 then the 320 probe.
run_cell B7_lazy_224  224  8  90
run_cell B8_lazy_320  320  8  90

# Aggregator
python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C9_MEAN = 0.0790
C9_SD   = 0.0105

rows.sort(key=lambda r: -r["mAP_50"])
print()
print(f"=== Phase 7 lazy-loader validation ===")
print(f"{'cell':16s} {'input':>5s} {'batch':>5s} {'epochs':>6s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s}")
for r in rows:
    print(f"{r['_cell']:16s} {r['_input']:>5d} {r['_batch']:>5d} {r['_epochs']:>6d} "
          f"{r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f}")
print()
print(f"C9 5-seed anchor: mean = {C9_MEAN:.4f}  pstdev = {C9_SD:.4f}")
print(f"                  band  [{C9_MEAN-C9_SD:.4f}, {C9_MEAN+C9_SD:.4f}]")
print()
for r in rows:
    cell = r["_cell"]
    m = r["mAP_50"]
    if cell == "B7_lazy_224":
        if C9_MEAN - C9_SD <= m <= C9_MEAN + C9_SD:
            print(f"  B7 (224, sanity): mAP={m:.4f}  ✓ IN BAND — refactor clean")
        else:
            print(f"  B7 (224, sanity): mAP={m:.4f}  ✗ OUT OF BAND — refactor regression")
    elif cell == "B8_lazy_320":
        if m > C9_MEAN + C9_SD:
            print(f"  B8 (320, probe):  mAP={m:.4f}  ✓ LIFT (+{m-C9_MEAN:+.4f} over C9) — resolution axis works")
        elif m > C9_MEAN - C9_SD:
            print(f"  B8 (320, probe):  mAP={m:.4f}  · in band — no lift but no regression at 320")
        else:
            print(f"  B8 (320, probe):  mAP={m:.4f}  ✗ below band — 320 hurts at this batch/epoch")
PY

# Flag OOMs
for cl in B7 B8; do
  log=$(ls $OUT_DIR/${cl}_*.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  if grep -q 'CUDA out of memory' "$log" 2>/dev/null; then
    echo "[orch] $(basename $log .log) hit OOM" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== Phase 7 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
