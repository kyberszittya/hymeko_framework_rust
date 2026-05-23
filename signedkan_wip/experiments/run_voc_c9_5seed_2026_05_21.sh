#!/usr/bin/env bash
# Stage D-3-BREAK-VALIDATE — 5-seed validation of C9 winner.
#
# C9 single-seed result (Phase 3, 2026-05-21 18:20 CEST):
#   mAP_50 = 0.1149  (2.03× over C8 5-seed 0.0552, 7.5× over D-3-bis)
# Recipe: lam_gate_neg=2.0, epochs=90, n_box_queries=6,
#         ResNet18-ImageNet, nodelet head, bce gate loss.
#
# Phase 3 confirmed:
#   - C9 falsifier: 0.1149 > 0.06 → "MORE EPOCHS HELP"
#   - C10 (n_q=4): 0.0504 → at C8 floor; n_q=6 stays
#   - C11 (λ=5): 0.0398 → λ=2 is the sweet spot
#
# This 5-seed run turns the single-seed 0.1149 into a publishable claim.
# Falsifier (built into aggregator):
#   - mean < 0.060 → seed luck (collapsed to C8 floor)
#   - sd   > 0.030 → brittle (allow ~30% relative noise; was 0.0146/0.0552
#                     = 26% for C8 5-seed, so 0.030 is the bound)
#   - mean ≥ 0.060 AND sd ≤ 0.030 → CONFIRMED LIFT OVER C8 5-SEED
#
# Wall: ~32 min/seed × 5 = ~160 min total.
# Memory: 16 GiB cgroups RSS cap.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_c9_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
JSONL="${OUT_DIR}/5seed.jsonl"
: > "$JSONL"

echo "=== Stage D-3-BREAK-VALIDATE — C9 5-seed ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight VOC / Gömb training (Phase 3 just
# finished but be defensive).
echo "[orch] waiting for GPU (other VOC / signedkan runs)..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke' | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_seed () {
  local seed="$1"
  local logf="${OUT_DIR}/seed${seed}.log"
  local jsonl_one="${OUT_DIR}/seed${seed}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] SEED ${seed} START" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_c9_5s_${STAMP}_s${seed}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs 90 --input-size 224 \
      --batch-size 8 --n-box-queries 6 \
      --lr 0.003 --seed "$seed" \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind nodelet \
      --lam-gate-neg 2.0 \
      --gate-loss-kind bce \
      --jsonl-out "$jsonl_one" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))

  if [ -s "$jsonl_one" ]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl_one")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
out = pathlib.Path("$JSONL")
with out.open("a") as f:
    for r in rows:
        r["_seed"] = $seed
        f.write(json.dumps(r) + "\n")
PY
  fi
  local m
  m=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$jsonl_one').read_text().splitlines() if l.strip()]
    last = rows[-1] if rows else {}
    m = last.get('mAP_50')
    print(f'{m:.4f}' if isinstance(m, float) else str(m))
except Exception as exc:
    print(f'err:{exc}')
" 2>&1)
  echo "[$(date -Is)] SEED ${seed} DONE rc=$rc mAP_50=$m elapsed=${elapsed}s" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
  run_seed "$SEED"
done

python - <<PY | tee -a "$LOG"
import json, pathlib, statistics
rows = [json.loads(l) for l in pathlib.Path("$JSONL").read_text().splitlines() if l.strip()]
maps = [r["mAP_50"] for r in rows if isinstance(r.get("mAP_50"), float)]
print()
print(f"=== C9 5-seed validation ===")
print(f"  n = {len(maps)}")
if maps:
    mean = statistics.mean(maps)
    sd = statistics.pstdev(maps) if len(maps) >= 2 else 0.0
    print(f"  mAP_50 mean = {mean:.4f}  pstdev = {sd:.4f}")
    print(f"  per-seed: {[f'{m:.4f}' for m in maps]}")
    print()

    C8_MEAN  = 0.0552  # C8 5-seed mean (validated 2026-05-21 14:02)
    PUB      = 0.0153  # published D-3-bis
    GATE     = 0.20    # visit gate

    # Falsifier checks
    if mean < 0.060:
        print(f"  FALSIFIED (seed luck): mean {mean:.4f} < 0.060")
        print(f"  The single-seed 0.1149 was outlier; collapsed to C8 floor.")
    elif sd > 0.030:
        print(f"  HIGH VARIANCE: sd {sd:.4f} > 0.030 (>54% of mean)")
        print(f"  90-epoch recipe is brittle; needs more regularisation.")
    else:
        lift_vs_c8  = mean - C8_MEAN
        lift_vs_pub = mean - PUB
        gap_to_gate = GATE - mean
        print(f"  CONFIRMED.")
        print(f"  vs C8 5-seed ({C8_MEAN:.4f}): {lift_vs_c8:+.4f} ({mean/C8_MEAN:.2f}×)")
        print(f"  vs published D-3-bis ({PUB:.4f}): {lift_vs_pub:+.4f} ({mean/PUB:.2f}×)")
        print(f"  visit-gate gap: {gap_to_gate:+.4f} (mean/gate = {mean/GATE:.2f})")
        # paired-z if all 5 seeds beat C8 5-seed individually
        c8_per_seed = [0.0358, 0.0753, 0.0434, 0.0673, 0.0540]
        # caution: paired comparison only meaningful with same training data
        # at matched seeds; here we just count wins.
        c9_sorted = sorted(maps)
        c8_sorted = sorted(c8_per_seed)
        wins = sum(1 for a, b in zip(c9_sorted, c8_sorted) if a > b)
        print(f"  ranked head-to-head vs C8 per-seed: {wins}/5 C9 wins")
PY

echo "" | tee -a "$LOG"
echo "=== Stage D-3-BREAK-VALIDATE (C9) DONE $(date -Is) ===" | tee -a "$LOG"
echo "  jsonl: $JSONL" | tee -a "$LOG"
