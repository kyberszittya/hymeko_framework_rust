#!/usr/bin/env bash
# Stage D-3-BREAK — push the D-3-bis ceiling (mAP_50 = 0.0153) on VOC2007.
#
# Per docs/plans/2026-05-21-voc-d3-break/plan.md (to be written morning).
# This is a *single-seed grid* on VOC2007 trainval (5011 images, nodelet
# head, ResNet18-ImageNet backbone). Goal: identify the cell that
# breaks 0.020 mAP_50 (clears the D-3-tris partial-win threshold).
#
# Cells (8):
#   C1  baseline           : lam_gate_neg=1.0  epochs=30 n_q=12  (D-3-bis replication)
#   C2  push-suppression   : lam_gate_neg=2.0  epochs=30 n_q=12
#   C3  hard-suppression   : lam_gate_neg=5.0  epochs=30 n_q=12
#   C4  longer-train       : lam_gate_neg=1.0  epochs=60 n_q=12
#   C5  longer+harder      : lam_gate_neg=2.0  epochs=60 n_q=12
#   C6  match-provisioning : lam_gate_neg=1.0  epochs=30 n_q=6  (closer to 2.4 GTs/img)
#   C7  match+long         : lam_gate_neg=1.0  epochs=60 n_q=6
#   C8  match+long+harder  : lam_gate_neg=2.0  epochs=60 n_q=6
#
# Wall budget: ~10-20 min/cell @ 30ep, ~25 min @ 60ep → ~2.5 h total.
# Memory budget: 16 GiB (cgroups RSS gate). RTX 2070 SUPER 8 GiB.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_d3_break_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK — VOC2007 grid ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight signedkan_wip / vision training.
echo "[orch] waiting for GPU (other signedkan_wip / vision runs)..." | tee -a "$LOG"
while pgrep -af 'signedkan_wip\.src\.(run_optuna_search|run_final_cell|run_gomb_smoke|vision\.train)' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_cell () {
  local cell="$1"; shift
  local epochs="$1"; shift
  local n_q="$1"; shift
  local lam_g="$1"; shift
  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START epochs=${epochs} n_q=${n_q} lam_gate_neg=${lam_g}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_d3_break_${STAMP}_${cell}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs "$epochs" --input-size 224 \
      --batch-size 8 --n-box-queries "$n_q" \
      --lr 0.003 --seed 0 \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind nodelet \
      --lam-gate-neg "$lam_g" \
      --gate-loss-kind bce \
      --jsonl-out "$jsonl" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))

  if [ -s "$jsonl" ]; then
    # Append cell label for aggregation
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
out = pathlib.Path("$GRID_JSONL")
with out.open("a") as f:
    for r in rows:
        r["_cell"] = "$cell"
        r["_epochs_cfg"] = $epochs
        r["_n_q"] = $n_q
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

# 8-cell grid
run_cell C1_lg1.0_e30_q12  30 12 1.0
run_cell C2_lg2.0_e30_q12  30 12 2.0
run_cell C3_lg5.0_e30_q12  30 12 5.0
run_cell C4_lg1.0_e60_q12  60 12 1.0
run_cell C5_lg2.0_e60_q12  60 12 2.0
run_cell C6_lg1.0_e30_q6   30  6 1.0
run_cell C7_lg1.0_e60_q6   60  6 1.0
run_cell C8_lg2.0_e60_q6   60  6 2.0

# Aggregate
python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]
rows.sort(key=lambda r: -r["mAP_50"])
print()
print(f"=== Grid summary ({len(rows)} cells, sorted by mAP_50) ===")
print(f"{'cell':32s} {'mAP_50':>8s} {'mIoU':>6s} {'cls_acc':>8s}")
for r in rows:
    print(f"{r.get('_cell', '?'):32s} {r['mAP_50']:>8.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('matched_cls_acc', float('nan')):>8.3f}")
if rows:
    best = rows[0]
    print()
    print(f"BEST: {best['_cell']} mAP_50={best['mAP_50']:.4f}")
    if best['mAP_50'] >= 0.020:
        print(f"PASS partial-win threshold (>=0.020) — queue 5-seed of best cell")
    else:
        print(f"BELOW 0.020 — head architecture limit; pivot to backbone or matcher")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-3-BREAK DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
