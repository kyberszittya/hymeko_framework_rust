#!/usr/bin/env bash
# Stage D-3-BREAK Phase 4 — input-resolution axis after C9 5-seed.
#
# Plan: docs/plans/2026-05-21-voc-input-resolution/plan.{tex,pdf,tikz,mmd}.
#
# C9 5-seed (2026-05-21 21:34 CEST) anchor: mAP_50 = 0.0790 ± 0.0105
# at 224×224, batch 8, 90 ep. Within-head tuning mapped: n_q=6 floor,
# λ=2 sweet spot, 90 ep on the steep descent.
#
# 3 cells (single-seed):
#   C12  320 px  batch 4  90 ep   (same epochs, 1.4× scale)
#   C13  320 px  batch 4  60 ep   (test if higher-res offsets shorter train)
#   C14  416 px  batch 2  60 ep   (8 GiB ceiling probe; may OOM)
#
# Falsifier verdicts embedded in the aggregator.
# Wall budget: ~4-5 h. Memory cap: 16 GiB cgroups RSS.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_input_resolution_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 4 — input resolution ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight VOC / Gömb training.
echo "[orch] waiting for GPU (other VOC / signedkan runs)..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke' | grep -v "$$" | grep -v "$0" | grep -q .; do
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
  echo "[$(date -Is)] CELL ${cell} START input=${input} batch=${batch} epochs=${epochs}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_inres_${STAMP}_${cell}.scope" \
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
        r["_cell"]     = "$cell"
        r["_input"]    = $input
        r["_batch"]    = $batch
        r["_epochs"]   = $epochs
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

run_cell C12_320_b4_e90  320 4 90
run_cell C13_320_b4_e60  320 4 60
run_cell C14_416_b2_e60  416 2 60

python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C9_MEAN = 0.0790  # C9 5-seed mean
C9_SD   = 0.0105  # C9 5-seed pstdev
PUB     = 0.0153  # published D-3-bis
GATE    = 0.20    # visit gate

rows.sort(key=lambda r: -r["mAP_50"])

print()
print(f"=== Phase 4 grid summary (sorted by mAP_50) ===")
print(f"{'cell':22s} {'input':>5s} {'batch':>5s} {'epochs':>6s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s} {'loss_end':>8s}")
for r in rows:
    print(f"{r['_cell']:22s} {r['_input']:>5d} {r['_batch']:>5d} "
          f"{r['_epochs']:>6d} {r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f} "
          f"{r.get('loss_end', float('nan')):>8.3f}")

print()
print(f"C9 5-seed anchor: mean = {C9_MEAN:.4f}  pstdev = {C9_SD:.4f}")
print(f"(within-noise band: [{C9_MEAN - C9_SD:.4f}, {C9_MEAN + C9_SD:.4f}])")
print()
print("=== Falsifier verdicts ===")
for r in rows:
    cell = r["_cell"]
    m = r["mAP_50"]
    if cell.startswith("C12"):
        if m > 0.085:
            v = f"RESOLUTION IS THE LEVER → 5-seed validate C12 (+{m-C9_MEAN:+.4f} vs C9)"
        elif m >= C9_MEAN - C9_SD:
            v = "directional flat — resolution alone doesn't dominate at this backbone size"
        else:
            v = f"resolution HURTS at this backbone — pivot to backbone/FPN"
        print(f"  C12 (320, 90 ep):  mAP={m:.4f}  → {v}")
    elif cell.startswith("C13"):
        # compare to C12 directly
        c12 = next((r for r in rows if r["_cell"].startswith("C12")), None)
        if c12 and abs(m - c12["mAP_50"]) <= 0.005:
            v = "60 ep is enough at 320 (save GPU on 5-seed)"
        elif c12 and m < c12["mAP_50"] - 0.005:
            v = "long training still matters at 320 (use C12 90-ep for 5-seed)"
        else:
            v = "C13 > C12 — 60 ep is actually better at 320 (overfitting at 90?)"
        print(f"  C13 (320, 60 ep):  mAP={m:.4f}  → {v}")
    elif cell.startswith("C14"):
        c12 = next((r for r in rows if r["_cell"].startswith("C12")), None)
        if c12 and m > c12["mAP_50"]:
            v = f"416 > 320 — retry C14 with smaller backbone (+{m-c12['mAP_50']:+.4f})"
        else:
            v = "416 hits diminishing returns or backbone limit"
        print(f"  C14 (416, 60 ep):  mAP={m:.4f}  → {v}")

if rows:
    best = rows[0]
    print()
    print(f"=== Recommendation ===")
    print(f"  Best Phase-4 cell: {best['_cell']}  mAP_50 = {best['mAP_50']:.4f}")
    print(f"  vs C9 5-seed: {best['mAP_50']-C9_MEAN:+.4f}  "
          f"(C9 within-noise band [{C9_MEAN-C9_SD:.4f}, {C9_MEAN+C9_SD:.4f}])")
    print(f"  vs visit gate (0.20): {GATE - best['mAP_50']:.4f} short ({best['mAP_50']/GATE:.2f} of gate)")
    if best['mAP_50'] > C9_MEAN + C9_SD:
        print(f"  ACTION: 5-seed validation of {best['_cell']}")
    elif best['mAP_50'] >= C9_MEAN - C9_SD:
        print(f"  ACTION: input-resolution is directional flat; pivot to backbone (ResNet18 → ResNet34/50)")
    else:
        print(f"  ACTION: input-resolution NEGATIVE at this backbone size; pivot to backbone")
PY

# Mark C14 as OOM-FAIL if its jsonl is empty.
for cell in C14_416_b2_e60; do
  if [ ! -s "${OUT_DIR}/${cell}.jsonl" ]; then
    echo "[orch] ${cell} produced no JSONL — likely CUDA OOM. Documenting as 8 GiB ceiling hit." | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== Stage D-3-BREAK Phase 4 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
