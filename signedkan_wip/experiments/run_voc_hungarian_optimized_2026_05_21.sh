#!/usr/bin/env bash
# Stage D-3-BREAK Phase 5 — optimized Hungarian head sweep.
#
# Methodology fix: the "nodelet > Hungarian" architectural claim from
# Stage D-3 was made at the UNOPTIMIZED recipe (n_q=12, 30 ep,
# lam_no_obj=1.0). After C9 5-seed (0.0790 ± 0.0105) showed recipe
# is the dominant lever (4-5× of the lift came from training length
# alone), the architectural claim is unjustified until Hungarian
# gets the same C9-grade recipe.
#
# Hungarian head has no per-query gate (the variable that
# --lam-gate-neg controls); its analog is --lam-no-obj (the
# no-object class weight). This sweep applies C9 levers
# (n_q=6, 90 ep, ResNet18-IN) and sweeps lam_no_obj.
#
# 6 cells (single-seed each):
#   H1  hungarian  n_q=6   90 ep   lam_no_obj=0.5  (baseline, C9 levers)
#   H2  hungarian  n_q=6   90 ep   lam_no_obj=2.0  (matched suppression)
#   H3  hungarian  n_q=6   90 ep   lam_no_obj=5.0  (D-2b's best at 30 ep, now at 90)
#   H4  hungarian  n_q=12  90 ep   lam_no_obj=2.0  (over-provisioning control)
#   H5  hungarian  n_q=6   60 ep   lam_no_obj=2.0  (epoch control)
#   H6  hungarian  n_q=4   90 ep   lam_no_obj=2.0  (tighter provisioning probe)
#
# Decisive comparison vs C9 5-seed (0.0790):
#   - Hungarian best > 0.079 → nodelet "architectural lift" claim FALSIFIED
#   - Hungarian best < 0.060 → nodelet IS architecturally better
#   - Hungarian best 0.060–0.079 → ambiguous (architecture vs noise)
#
# Wall: ~30 min/cell × 6 = ~180 min = ~3 h.
# Memory cap: 16 GiB cgroups RSS.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_hungarian_optimized_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 5 — optimized Hungarian ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight VOC / signedkan training.
echo "[orch] waiting for GPU (other VOC / signedkan runs)..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke' | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_cell () {
  local cell="$1"; shift
  local n_q="$1"; shift
  local epochs="$1"; shift
  local lam_no_obj="$1"; shift
  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START n_q=${n_q} epochs=${epochs} lam_no_obj=${lam_no_obj}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_hu_${STAMP}_${cell}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs "$epochs" --input-size 224 \
      --batch-size 8 --n-box-queries "$n_q" \
      --lr 0.003 --seed 0 \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind hungarian \
      --lam-no-obj "$lam_no_obj" \
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
        r["_n_q"]        = $n_q
        r["_epochs"]     = $epochs
        r["_lam_no_obj"] = $lam_no_obj
        r["_head"]       = "hungarian"
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

run_cell H1_q6_e90_lno0.5  6  90 0.5
run_cell H2_q6_e90_lno2.0  6  90 2.0
run_cell H3_q6_e90_lno5.0  6  90 5.0
run_cell H4_q12_e90_lno2.0 12 90 2.0
run_cell H5_q6_e60_lno2.0  6  60 2.0
run_cell H6_q4_e90_lno2.0  4  90 2.0

python - <<PY | tee -a "$LOG"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C9_MEAN = 0.0790  # C9 5-seed nodelet mean
C9_SD   = 0.0105
PUB     = 0.0153  # original published D-3-bis (hungarian, unoptimized)
GATE    = 0.20

rows.sort(key=lambda r: -r["mAP_50"])

print()
print(f"=== Phase 5 Hungarian grid (sorted by mAP_50) ===")
print(f"{'cell':22s} {'nq':>3s} {'ep':>3s} {'lno':>4s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s} {'loss_end':>8s}")
for r in rows:
    print(f"{r['_cell']:22s} {r['_n_q']:>3d} {r['_epochs']:>3d} "
          f"{r['_lam_no_obj']:>4.1f} {r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f} "
          f"{r.get('loss_end', float('nan')):>8.3f}")

print()
print(f"Comparison anchors:")
print(f"  Original D-3-bis (hungarian unoptimised): mAP = {PUB:.4f}")
print(f"  C9 5-seed (nodelet at C9 recipe):         mAP = {C9_MEAN:.4f} ± {C9_SD:.4f}")
print(f"                                            band  [{C9_MEAN-C9_SD:.4f}, {C9_MEAN+C9_SD:.4f}]")
print()

if rows:
    best = rows[0]
    bm = best["mAP_50"]
    print(f"=== Architectural verdict ===")
    print(f"  Best Hungarian cell: {best['_cell']}  mAP_50 = {bm:.4f}")
    print(f"  Lift over unoptimised Hungarian: {bm-PUB:+.4f} ({bm/PUB:.2f}×)  ← recipe-only lift")
    print(f"  vs C9 nodelet 5-seed mean:        {bm-C9_MEAN:+.4f}")
    if bm > C9_MEAN + C9_SD:
        print(f"  VERDICT: Nodelet 'architectural lift' FALSIFIED.")
        print(f"           Hungarian at C9 recipe BEATS nodelet at C9 recipe.")
        print(f"           The 5.16× lift was recipe-only; the head choice doesn't matter.")
        print(f"  ACTION:  5-seed validate the Hungarian best cell.")
    elif bm >= C9_MEAN - C9_SD:
        print(f"  VERDICT: AMBIGUOUS — Hungarian and nodelet are within noise at matched recipe.")
        print(f"           Architecture lift is at most σ-sized.")
        print(f"  ACTION:  Run 5-seed of best Hungarian + paired test vs C9 5-seed.")
    else:
        gap = C9_MEAN - bm
        print(f"  VERDICT: Nodelet IS architecturally better at C9 recipe.")
        print(f"           Hungarian best lands {gap:.4f} below the C9 band.")
        print(f"  ACTION:  The 'nodelet > hungarian' claim holds; document the matched-recipe comparison.")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-3-BREAK Phase 5 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
