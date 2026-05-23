#!/usr/bin/env bash
# Stage D-1 — HyMeYOLO with ImageNet-pretrained ResNet18 backbone on VOC2007.
# Plan: docs/plans/2026-05-18-hymeyolo-stage-d1-pretrain/.
#
# Rollback step from the falsified Stage D (smoke mAP_50 = 0.0073).
#
# Two-stage launch per CLAUDE §3:
#   1. PRODUCTION-SCALE SMOKE (1 seed × 5011 img × 30 epochs).
#      Falsifier gate: mAP_50 < 0.05 → backbone is NOT the bottleneck.
#      Production gate: mAP_50 ≥ 0.10 needed to queue the 5-seed.
#      Budget: ~75 min GPU.
#
#   2. 5-SEED HEADLINE (5 seeds × 80 epochs).
#      Falsifier zones:
#        < 0.20  → architectural problem deeper than backbone capacity.
#        0.20-0.30 → recipe-needs-work; open D-2 (augmentation).
#        ≥ 0.30  → PASS.
#      Budget: ~7.5 h GPU.
#
# Queues behind any in-flight signedkan_wip training.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/stage_d1_voc2007_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
SMOKE_JSONL="${OUT_DIR}/smoke_seed0.jsonl"
FIVE_SEED_JSONL="${OUT_DIR}/5seed.jsonl"

echo "=== Stage D-1 — HyMeYOLO with ImageNet-pretrained ResNet18 ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight signedkan_wip training (including the
# IMDB arch-fairness run launched alongside this orchestrator).
echo "[orch] waiting for GPU..." | tee -a "$LOG"
while pgrep -af 'signedkan_wip\.src\.(run_optuna_search|run_final_cell|run_gomb_smoke|vision\.train|sequence\.(train|run))' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

# ─── Stage 1: production-scale smoke ──────────────────────────────────

echo "" | tee -a "$LOG"
echo "## Stage 1 — production-scale smoke (1 seed × 5011 img × 30 epochs)" | tee -a "$LOG"

SMOKE_LOG="${OUT_DIR}/smoke_seed0.log"
t0=$(date +%s)
echo "[$(date -Is)] SMOKE START seed=0" | tee -a "$LOG"

systemd-run --user --scope --quiet \
  --unit="stage_d1_voc_smoke_${STAMP}.scope" \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  python -u -m signedkan_wip.src.vision.train_voc_stagec \
    --image-set trainval --epochs 30 --input-size 224 \
    --batch-size 8 --n-box-queries 12 \
    --backbone resnet18_imagenet \
    --lr 0.003 --seed 0 \
    --device cuda \
    --save-checkpoint "$OUT_DIR/checkpoints" \
    --jsonl-out "$SMOKE_JSONL" \
  > "$SMOKE_LOG" 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
echo "[$(date -Is)] SMOKE DONE rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"

SMOKE_MAP=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$SMOKE_JSONL').read_text().splitlines() if l.strip()]
    if rows:
        m = rows[-1].get('mAP_50')
        if isinstance(m, (int, float)): print(f'{m:.4f}')
        else: print('null')
    else:
        print('empty')
except FileNotFoundError:
    print('missing')
" 2>&1)
echo "[orch] smoke mAP_50=$SMOKE_MAP" | tee -a "$LOG"

# Two gates per Stage D-1 plan §7:
#   < 0.05  → backbone is NOT the bottleneck; deeper falsification, no 5-seed.
#   0.05-0.10 → near-miss; abort 5-seed and flag for diagnostic.
#   ≥ 0.10  → clear; queue 5-seed.
if [ "$SMOKE_MAP" = "null" ] || [ "$SMOKE_MAP" = "empty" ] || [ "$SMOKE_MAP" = "missing" ]; then
  echo "[orch] SMOKE PRODUCED NO mAP — aborting 5-seed launch" | tee -a "$LOG"
  exit 1
fi

zone=$(python -c "
v = float('$SMOKE_MAP')
print('below_floor' if v < 0.05 else ('near_miss' if v < 0.10 else 'pass'))
" 2>&1)
echo "[orch] smoke zone=$zone" | tee -a "$LOG"

if [ "$zone" = "below_floor" ]; then
  echo "[orch] SMOKE mAP_50=$SMOKE_MAP < 0.05 — STAGE D-1 FALSIFIED" | tee -a "$LOG"
  echo "[orch] backbone is NOT the bottleneck; do not queue 5-seed." | tee -a "$LOG"
  echo "[orch] Per plan §7: open Stage D-2 (augmentation) or" | tee -a "$LOG"
  echo "[orch] examine Hungarian head / loss formulation." | tee -a "$LOG"
  exit 2
fi
if [ "$zone" = "near_miss" ]; then
  echo "[orch] SMOKE mAP_50=$SMOKE_MAP in [0.05, 0.10) — NEAR-MISS" | tee -a "$LOG"
  echo "[orch] Aborting 5-seed launch pending diagnostic." | tee -a "$LOG"
  echo "[orch] Per plan §5: run 60-epoch diagnostic before queuing 5-seed." | tee -a "$LOG"
  exit 3
fi

echo "[orch] SMOKE PASSES gate (mAP_50=$SMOKE_MAP >= 0.10), proceeding to 5-seed" | tee -a "$LOG"

# ─── Stage 2: 5-seed headline run ────────────────────────────────────

echo "" | tee -a "$LOG"
echo "## Stage 2 — 5-seed headline (5 × 5011 img × 80 epochs)" | tee -a "$LOG"
: > "$FIVE_SEED_JSONL"

run_one() {
  local seed="$1"
  local logf="${OUT_DIR}/5seed_seed${seed}.log"
  local jsonl="${OUT_DIR}/5seed_seed${seed}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] 5SEED START seed=$seed" | tee -a "$LOG"
  systemd-run --user --scope --quiet \
    --unit="stage_d1_voc_5seed_${STAMP}_s${seed}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -u -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs 80 --input-size 224 \
      --batch-size 8 --n-box-queries 12 \
      --backbone resnet18_imagenet \
      --lr 0.003 --seed "$seed" \
      --device cuda \
      --save-checkpoint "$OUT_DIR/checkpoints" \
      --jsonl-out "$jsonl" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ -s "$jsonl" ]; then
    cat "$jsonl" >> "$FIVE_SEED_JSONL"
  fi
  local m
  m=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$jsonl').read_text().splitlines() if l.strip()]
    v = rows[-1].get('mAP_50') if rows else None
    print(f'{v:.4f}' if isinstance(v, (int, float)) else 'null')
except Exception:
    print('error')
" 2>&1)
  echo "[$(date -Is)] 5SEED DONE  seed=$seed rc=$rc mAP_50=$m elapsed=${elapsed}s" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
  run_one "$SEED"
done

# Aggregate
python - <<PY | tee -a "$LOG"
import json, statistics, pathlib
rows = [json.loads(l) for l in pathlib.Path("$FIVE_SEED_JSONL").read_text().splitlines() if l.strip()]
maps = [r["mAP_50"] for r in rows if isinstance(r.get("mAP_50"), (int, float))]
print(f"[5-seed aggregate] n={len(maps)}")
if maps:
    mean = statistics.mean(maps)
    sd = statistics.pstdev(maps) if len(maps) >= 2 else 0.0
    print(f"  mAP_50 mean = {mean:.4f}  pstdev = {sd:.4f}")
    print(f"  per-seed: {[f'{m:.4f}' for m in maps]}")
    if mean < 0.20:
        print(f"  FALSIFIER: 5-seed mean < 0.20 → architectural problem deeper than backbone")
    elif mean < 0.30:
        print(f"  RECIPE-NEEDS-WORK zone (0.20 <= mean < 0.30) — open Stage D-2 augmentation")
    else:
        print(f"  PASS: mean >= 0.30")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-1 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  smoke jsonl: $SMOKE_JSONL" | tee -a "$LOG"
echo "  5-seed jsonl: $FIVE_SEED_JSONL" | tee -a "$LOG"
