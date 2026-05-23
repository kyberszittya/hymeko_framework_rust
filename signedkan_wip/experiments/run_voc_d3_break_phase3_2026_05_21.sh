#!/usr/bin/env bash
# Stage D-3-BREAK Phase 3 — push the C8 ceiling.
#
# Plan: docs/plans/2026-05-21-voc-d3-break-phase3/plan.{tex,pdf,tikz,mmd}.
#
# C8 5-seed validated (2026-05-21 14:02): mAP_50 = 0.0552 ± 0.0146,
# 3.6× over published D-3-bis (0.0153). Diagnostic: loss still
# descending at ep 60 (slope -0.245/10ep), best in the grid.
#
# 3 cells (single-seed each):
#   C9   longer train       λ=2.0  90 ep  n_q=6
#   C10  harder provisioning λ=2.0  60 ep  n_q=4
#   C11  harder suppression λ=5.0  60 ep  n_q=6
#
# Falsifier tests per cell embedded in the aggregator below.
# Wall budget: ~75-85 min. Memory cap: 16 GiB cgroups RSS.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_d3_break_phase3_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 3 ===" | tee -a "$LOG"
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
  local epochs="$1"; shift
  local n_q="$1"; shift
  local lam_g="$1"; shift
  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START epochs=${epochs} n_q=${n_q} lam_gate_neg=${lam_g}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_d3_break_p3_${STAMP}_${cell}.scope" \
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
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
out = pathlib.Path("$GRID_JSONL")
with out.open("a") as f:
    for r in rows:
        r["_cell"]         = "$cell"
        r["_epochs_cfg"]   = $epochs
        r["_n_q"]          = $n_q
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

run_cell C9_lg2.0_e90_q6   90 6 2.0
run_cell C10_lg2.0_e60_q4  60 4 2.0
run_cell C11_lg5.0_e60_q6  60 6 5.0

python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C8_BASELINE = 0.0552  # C8 5-seed mean
PUB         = 0.0153  # published D-3-bis

rows.sort(key=lambda r: -r["mAP_50"])

print()
print(f"=== Phase 3 grid summary (sorted by mAP_50) ===")
print(f"{'cell':24s} {'ep':>3s} {'nq':>3s} {'lam':>4s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s} {'loss_end':>8s} {'drop%':>6s}")
for r in rows:
    print(f"{r['_cell']:24s} {r['_epochs_cfg']:>3d} {r['_n_q']:>3d} "
          f"{r['_lam_gate_neg']:>4.1f} {r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f} "
          f"{r.get('loss_end', float('nan')):>8.3f} "
          f"{r.get('loss_drop_pct', float('nan')):>6.1f}")

print()
print("=== Falsifier verdicts ===")
for r in rows:
    cell = r["_cell"]
    m = r["mAP_50"]
    if cell.startswith("C9"):
        verdict = "MORE EPOCHS HELP"  if m > 0.06 else ("at C8 floor — 60 ep is enough" if m >= C8_BASELINE - 0.01 else "LONGER TRAIN HURTS (overfit?)")
        print(f"  C9 (90 ep):   mAP={m:.4f}  → {verdict}")
    elif cell.startswith("C10"):
        verdict = "PROVISIONING FLOOR FURTHER" if m > 0.055 else ("at C8 floor" if m >= C8_BASELINE - 0.01 else "TOO FEW QUERIES — provisioning at floor")
        print(f"  C10 (n_q=4):  mAP={m:.4f}  → {verdict}")
    elif cell.startswith("C11"):
        verdict = "λ=5 SWEET SPOT" if m > 0.055 else ("at C8 floor" if m >= C8_BASELINE - 0.01 else "λ=2 is the sweet spot — Phase-1 C3 confirmed")
        print(f"  C11 (λ=5):    mAP={m:.4f}  → {verdict}")

if rows:
    best = rows[0]
    print()
    print(f"=== Recommendation ===")
    print(f"  Best Phase-3 cell: {best['_cell']}  mAP_50 = {best['mAP_50']:.4f}")
    if best['mAP_50'] > C8_BASELINE + 0.005:
        print(f"  ACTION: 5-seed validation of {best['_cell']} (lift {best['mAP_50']-C8_BASELINE:+.4f} over C8)")
    else:
        print(f"  ACTION: C8 stays the recipe; pivot lever (e.g., backbone, FPN, anchor init)")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-3-BREAK Phase 3 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
