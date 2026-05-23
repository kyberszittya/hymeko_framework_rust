#!/usr/bin/env bash
# Stage D — HyMeYOLO Stage C on PASCAL VOC2007.
#
# Per plan: docs/plans/2026-05-17-hymeyolo-stage-d-pascal-voc/plan.tex
#
# Two-stage launch:
#   1. PRODUCTION-SCALE SMOKE (CLAUDE §3): 1 seed × VOC2007 trainval
#      (5011 images) × 30 epochs × input_size=224. Blocks the 5-seed
#      launch if mAP_50 < 0.10 (chance is ~0 on 20 classes).
#      Budget: ~60-90 min GPU.
#
#   2. 5-SEED HEADLINE: 5 seeds × VOC2007 trainval × 80 epochs.
#      Falsifier: 5-seed mean test mAP_50 < 0.20 → architecture does
#      not transfer to natural images; do NOT chain to Stage E.
#      Budget: ~6h GPU.
#
# Queues behind any in-flight signedkan_wip training (HSiKAN rescore,
# Gömb shuffle audit, etc.) via pgrep loop.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/stage_d_voc2007_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
SMOKE_JSONL="${OUT_DIR}/smoke_seed0.jsonl"
FIVE_SEED_JSONL="${OUT_DIR}/5seed.jsonl"

echo "=== Stage D — HyMeYOLO Stage C on VOC2007 ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight HSiKAN / Gömb / VOC training.
echo "[orch] waiting for GPU (signedkan_wip training)..." | tee -a "$LOG"
while pgrep -af 'signedkan_wip\.src\.(run_optuna_search|run_final_cell|run_gomb_smoke|vision\.train)' \
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
  --unit="stage_d_voc_smoke_${STAMP}.scope" \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  python -m signedkan_wip.src.vision.train_voc_stagec \
    --image-set trainval --epochs 30 --input-size 224 \
    --batch-size 8 --n-box-queries 12 \
    --lr 0.003 --seed 0 \
    --device cuda \
    --save-checkpoint "$OUT_DIR/checkpoints" \
    --jsonl-out "$SMOKE_JSONL" \
  > "$SMOKE_LOG" 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
echo "[$(date -Is)] SMOKE DONE rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"

# Check the smoke result before queuing 5-seed.
SMOKE_MAP=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$SMOKE_JSONL').read_text().splitlines() if l.strip()]
    if rows:
        m = rows[-1].get('mAP_50')
        if m is None: print('null')
        else: print(f'{m:.4f}')
    else:
        print('empty')
except FileNotFoundError:
    print('missing')
" 2>&1)
echo "[orch] smoke mAP_50=$SMOKE_MAP" | tee -a "$LOG"

# Block 5-seed if smoke is below the 0.10 floor.
if [ "$SMOKE_MAP" = "null" ] || [ "$SMOKE_MAP" = "empty" ] || [ "$SMOKE_MAP" = "missing" ]; then
  echo "[orch] SMOKE PRODUCED NO mAP — aborting 5-seed launch" | tee -a "$LOG"
  exit 1
fi

below_floor=$(python -c "print(1 if float('$SMOKE_MAP') < 0.10 else 0)" 2>&1)
if [ "$below_floor" = "1" ]; then
  echo "[orch] SMOKE mAP_50=$SMOKE_MAP < 0.10 — aborting 5-seed launch (CLAUDE §3 production-scale gate)" | tee -a "$LOG"
  exit 2
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
    --unit="stage_d_voc_5seed_${STAMP}_s${seed}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs 80 --input-size 224 \
      --batch-size 8 --n-box-queries 12 \
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
    print(f\"{rows[-1].get('mAP_50', 'null'):.4f}\" if rows and isinstance(rows[-1].get('mAP_50'), float) else 'null')
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
        print(f"  FALSIFIER HIT: 5-seed mean < 0.20 — Stage D plan rollback path active")
    elif mean < 0.30:
        print(f"  RECIPE-NEEDS-WORK zone (0.20 <= mean < 0.30) — open Stage D-1")
    else:
        print(f"  PASS: mean >= 0.30")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D DONE $(date -Is) ===" | tee -a "$LOG"
echo "  smoke jsonl: $SMOKE_JSONL" | tee -a "$LOG"
echo "  5-seed jsonl: $FIVE_SEED_JSONL" | tee -a "$LOG"
