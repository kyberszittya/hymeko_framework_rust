#!/usr/bin/env bash
# Stage D-3-BREAK-PHASE-2 — quadtree-driven warmstart on VOC2007.
#
# Tests whether the curvature-aware AdaptiveQuadtreeRust leaf centres
# (variance + Forman κ) are better query priors than the saliency
# FPS baseline.  The Stage A-1 saliency-warmstart lever delivered
# +0.124 mAP_50 paired Δ on Cluttered MNIST (4.68σ); the open
# question is whether a multi-scale, curvature-aware variant
# transfers to natural images.
#
# 4-cell single-seed sweep across the 2 × 2 grid:
#   (warmstart-mode ∈ {off, quadtree}) × (lam_gate_neg ∈ {1.0, 2.0})
# All cells run with the locked D-3-bis recipe:
#   nodelet head, ResNet18-ImageNet backbone, 12 queries, 60 epochs,
#   bce gate loss, cosine schedule with 10-epoch warmup.
#
# Queues behind voc_d3_break_*.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_quadtree_warmstart_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3 PHASE 2 — quadtree warmstart ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

echo "[orch] waiting for any other VOC training to finish..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec' | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 60
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_cell () {
  local cell="$1"; shift
  local mode="$1"; shift     # off | quadtree
  local lam_g="$1"; shift
  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START mode=${mode} lam_gate_neg=${lam_g}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_qtw_${STAMP}_${cell}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs 60 --input-size 224 \
      --batch-size 8 --n-box-queries 12 \
      --lr 0.003 --seed 0 \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind nodelet \
      --lam-gate-neg "$lam_g" \
      --gate-loss-kind bce \
      --warmstart-mode "$mode" \
      --warmstart-bootstrap-n 128 \
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
        r["_cell"] = "$cell"
        r["_warmstart"] = "$mode"
        r["_lam_gate_neg"] = $lam_g
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

# 4-cell grid
run_cell P1_off_lg1.0       off      1.0
run_cell P2_off_lg2.0       off      2.0
run_cell P3_quadtree_lg1.0  quadtree 1.0
run_cell P4_quadtree_lg2.0  quadtree 2.0

python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]
rows.sort(key=lambda r: -r["mAP_50"])
print()
print(f"=== Phase-2 grid summary ({len(rows)} cells) ===")
print(f"{'cell':20s} {'warmstart':>10s} {'lam_g':>6s} {'mAP_50':>8s} {'mIoU':>6s}")
for r in rows:
    print(f"{r.get('_cell','?'):20s} {r.get('_warmstart','?'):>10s} "
          f"{r.get('_lam_gate_neg','-'):>6} "
          f"{r['mAP_50']:>8.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f}")
# Paired Δ: quadtree vs off at each lam_g
for lg in (1.0, 2.0):
    off  = [r for r in rows if r.get('_warmstart')=='off' and r.get('_lam_gate_neg')==lg]
    qtr  = [r for r in rows if r.get('_warmstart')=='quadtree' and r.get('_lam_gate_neg')==lg]
    if off and qtr:
        d = qtr[0]['mAP_50'] - off[0]['mAP_50']
        print(f"[Δ] lam_g={lg}  quadtree-off = {d:+.4f} mAP_50")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-3 Phase-2 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
