#!/usr/bin/env bash
# Stage D-3-BREAK Phase 6 — backbone-shrink × resolution sweep.
#
# Why: C9 5-seed sits at 0.0790 mAP_50 (5.16× over D-3-bis).  Phase 4
# (input-resolution at ResNet18-IN) OOMed at 8 GiB — every cell crashed.
# The natural next axis is *shrink the backbone, then revisit higher
# resolution*.  Two small backbones are available out of the box:
#
#   resnet                — custom 3-layer ResNet from scratch (~107k params)
#   hsikan                — HSiKAN-CR (Catmull-Rom basis) (~136k params)
#   resnet18_imagenet     — the C9 baseline (11.7M, ImageNet-pretrained)
#
# Smaller activation footprints should let us run 320 px without OOM.
#
# 6 single-seed cells; falsifier = any cell beats C9 5-seed mean
# 0.0790 ± 0.0105 → 5-seed validate tomorrow.
#
# Wall: ~25-35 min/cell × 6 = ~3 h.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_backbone_shrink_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 6 — backbone shrink ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

echo "[orch] waiting for GPU..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke|ros2 launch hymeko_ros2_demo' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_cell () {
  local cell="$1"; shift
  local backbone="$1"; shift
  local input="$1"; shift
  local batch="$1"; shift
  local epochs="$1"; shift
  local checkpoint="$1"; shift  # "true" or "false"

  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START bb=${backbone} input=${input} batch=${batch} ep=${epochs} ckpt=${checkpoint}" | tee -a "$LOG"

  local ckpt_flag=""
  if [ "$checkpoint" = "true" ]; then ckpt_flag="--backbone-checkpoint"; fi

  systemd-run --user --scope --quiet \
    --unit="voc_bb_${STAMP}_${cell}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs "$epochs" --input-size "$input" \
      --batch-size "$batch" --n-box-queries 6 \
      --lr 0.003 --seed 0 \
      --device cuda \
      --backbone "$backbone" \
      --query-head-kind nodelet \
      --lam-gate-neg 2.0 \
      --gate-loss-kind bce \
      $ckpt_flag \
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
        r["_cell"]       = "$cell"
        r["_backbone"]   = "$backbone"
        r["_input"]      = $input
        r["_batch"]      = $batch
        r["_epochs"]     = $epochs
        r["_ckpt"]       = "$checkpoint"
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

# 6 cells — smallest backbones first so we get early signal even on partial runs.
#                          cell                 backbone               input  batch  epochs  ckpt
run_cell B1_resnet_224         resnet                  224    8    90    false
run_cell B2_resnet_320         resnet                  320    8    90    false
run_cell B3_resnet_320_bs16    resnet                  320   16    90    false
run_cell B4_hsikan_224_ckpt    hsikan                  224    8    90    true
run_cell B5_hsikan_320_ckpt    hsikan                  320    4    90    true
run_cell B6_r18in_224_ref      resnet18_imagenet       224    8    90    false

# Aggregator
python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C9_MEAN = 0.0790
C9_SD   = 0.0105
PUB     = 0.0153

rows.sort(key=lambda r: -r["mAP_50"])

print()
print(f"=== Phase 6 backbone-shrink grid (sorted by mAP_50) ===")
print(f"{'cell':24s} {'backbone':18s} {'in':>3s} {'bs':>3s} {'ep':>3s} {'ckpt':>5s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s} {'loss_end':>8s}")
for r in rows:
    print(f"{r['_cell']:24s} {r['_backbone']:18s} "
          f"{r['_input']:>3d} {r['_batch']:>3d} {r['_epochs']:>3d} {r['_ckpt']:>5s} "
          f"{r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f} "
          f"{r.get('loss_end', float('nan')):>8.3f}")

print()
print(f"C9 5-seed anchor: mean = {C9_MEAN:.4f}  pstdev = {C9_SD:.4f}  band [{C9_MEAN-C9_SD:.4f}, {C9_MEAN+C9_SD:.4f}]")
print()
print("=== Falsifier verdicts ===")
if rows:
    best = rows[0]
    bm = best["mAP_50"]
    print(f"Best cell:  {best['_cell']}  mAP_50 = {bm:.4f}")
    if bm > C9_MEAN + C9_SD:
        print(f"  ACTION: 5-seed validate {best['_cell']} (lift {bm-C9_MEAN:+.4f} over C9)")
    elif bm >= C9_MEAN - C9_SD:
        print(f"  ACTION: backbone shrink is in C9 noise band; combine with other axis")
    else:
        print(f"  ACTION: backbone shrink NEGATIVE at this recipe; revisit input-res with grad-checkpoint")
PY

# Mark OOMs explicitly
for cell_log in $OUT_DIR/*.log; do
  if grep -q 'CUDA out of memory\|MemoryError' "$cell_log" 2>/dev/null; then
    cell=$(basename "$cell_log" .log)
    echo "[orch] $cell hit OOM — documented" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== Phase 6 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
