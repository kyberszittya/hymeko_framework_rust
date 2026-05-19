#!/usr/bin/env bash
# HymeYOLO Ricci × k-cycles 5-seed — validate the bug-fixed model.
#
# Context: the +kcycle config in the 2026-05-13 5-seed stage-7 results had
# a known localization bug — the signed-cycle aggregator was wired into
# the classification head but NOT into offset prediction. Result: low,
# low-variance mAP (0.204 ± 0.028). The bug-fixed RicciKCycleHyMeYOLOMulti
# wires the aggregator into BOTH offset AND cls.
#
# Goal: produce 5-seed mAP50 / mAP50:95 / mIoU on Cluttered MNIST for
# the +ricci+kcycle config, comparable to the 2026-05-13 stage-7
# results:
#   - boxes+circles:        0.715 ± 0.163 mAP50
#   - +ricci-mod:           0.723 ± 0.180 mAP50
#   - +kcycle (broken):     0.204 ± 0.028 mAP50
#   - +ricci+kcycle (NEW):  TBD ← this run
#
# This is the vision side of the "hypergraph revolution" — signed-cycle
# σ-products + Ricci curvature applied to multi-object detection on
# cluttered backgrounds.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/hymeyolo_ricci_kcycle_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== HymeYOLO +ricci+kcycle 5-seed validation ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Wait for any other Gömb / vision train job to finish first.
echo "[$(date -Is)] Waiting for in-flight Gömb / vision train to clear..." | tee -a "$LOG"
while pgrep -f "run_gomb_smoke|run_gomb_strict_benchmark|run_gomb_epinions_finetune|train_circles_ricci" 2>/dev/null \
        | grep -v $$ | grep -q .; do
    sleep 30
done
echo "[$(date -Is)] GPU clear. Beginning HymeYOLO 5-seed." | tee -a "$LOG"

# 5 seeds × 5000 images × 50 epochs (matches stage-7 protocol).
N_IMAGES=5000
EPOCHS=50

for seed in 0 1 2 3 4; do
    outf="${OUT_DIR}/seed${seed}.log"
    jsonl="${OUT_DIR}/seed${seed}.jsonl"
    echo "[$(date -Is)] START seed=$seed n_images=$N_IMAGES epochs=$EPOCHS" \
        | tee -a "$LOG"
    t0=$(date +%s)
    python -m signedkan_wip.src.vision.train_circles_ricci \
        --no-warm-start --schedule constant --warmup-epochs 0 \
        --n-images "$N_IMAGES" --epochs "$EPOCHS" --seed "$seed" \
        --device cuda \
        --configs '+ricci+kcycle' \
        --jsonl-out "$jsonl" \
        > "$outf" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    # Extract headline mAP50.
    map50=$(grep -E "mAP50=[0-9]" "$outf" | tail -1 | grep -oE "mAP50=[0-9.]+" | head -1)
    echo "[$(date -Is)] DONE  seed=$seed rc=$rc elapsed=${elapsed}s $map50" \
        | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# =================================================================
# Final aggregate
# =================================================================
echo "" | tee -a "$LOG"
echo "=== HymeYOLO 5-seed complete: $(date -Is) ===" | tee -a "$LOG"

python3 << EOF | tee -a "$LOG"
import json, glob, statistics
from pathlib import Path

out_dir = Path("$OUT_DIR")
rows = []
for f in sorted(out_dir.glob("seed*.jsonl")):
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

ricci_kcycle = [r for r in rows if r.get("config") == "+ricci+kcycle"]
print(f"\n+ricci+kcycle  n={len(ricci_kcycle)}")
if ricci_kcycle:
    for metric in ("mAP50", "mAP50_95", "mIoU", "box_acc", "circ_acc"):
        vals = [r.get(metric) for r in ricci_kcycle if isinstance(r.get(metric), (int, float))]
        if vals:
            m = statistics.mean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"  {metric:10}  {m:.4f} ± {s:.4f}  (n={len(vals)})")

print()
print("Reference (stage-7 2026-05-13 5-seed):")
print("  boxes+circles      mAP50 = 0.715 ± 0.163")
print("  +ricci-mod         mAP50 = 0.723 ± 0.180")
print("  +kcycle (broken)   mAP50 = 0.204 ± 0.028")
EOF

echo "Results in: $OUT_DIR" | tee -a "$LOG"
