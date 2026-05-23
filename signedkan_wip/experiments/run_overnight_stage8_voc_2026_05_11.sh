#!/bin/bash
# Stage 8 (2026-05-11): HyMeYOLO on Pascal VOC 2007.
#
# First properly-comparable detection benchmark for HyMeYOLO. Same
# 5-config ablation as Cluttered MNIST (baseline / boxes-only /
# circles-only / boxes+circles / +ricci-mod), now at 20 classes on
# real images. Trains on VOC2007 trainval (5011 images, 128x128).
#
# Reports mAP@0.5 + mAP@0.5:0.95 + mean matched IoU per config.
# The vanilla DETR-MNIST analogue (`baseline` HyMeYOLOMulti) is the
# fair comparator inside this script; for context, YOLOv5n on VOC2007
# trainval+test typically reports mAP@0.5 ≈ 0.45-0.55 at much larger
# input + many more epochs. We are running 30 epochs at 128x128 as a
# first measurement; expectation: 0.05-0.20 mAP@0.5 on +ricci-mod.
#
# Per CLAUDE.md §4: 16 GB cgroup cap per run.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage8
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-8 VOC2007 HyMeYOLO START ===" > "$MASTER"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] git SHA $(git rev-parse HEAD)" >> "$MASTER"
fi

run() {
    local name="$1"; shift
    if [ -s "$LOG/$name.jsonl" ] && grep -q '"mAP_50"' "$LOG/$name.jsonl" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name (already complete)" | tee -a "$MASTER"
        return 0
    fi
    local start=$(date +%s)
    echo "[$(date '+%H:%M:%S')] START $name" | tee -a "$MASTER"
    local timeout_s="${TIMEOUT_S:-10800}"
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope --quiet \
            -p MemoryMax=16G -p MemorySwapMax=0 \
            timeout "$timeout_s" "$@" > "$LOG/$name.stdout" 2> "$LOG/$name.err"
    else
        timeout "$timeout_s" "$@" > "$LOG/$name.stdout" 2> "$LOG/$name.err"
    fi
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ $rc -eq 0 ]; then
        local best
        best=$(python3 -c "
import json, sys
try:
    rows = [json.loads(l) for l in open('$LOG/$name.jsonl')]
    rows.sort(key=lambda r: -(r.get('mAP_50', 0) or 0))
    if rows:
        r = rows[0]
        print(f\"best={r['label']} mAP50={r.get('mAP_50',0):.3f} mAP50:95={r.get('mAP_50_95',0):.3f}\")
    else:
        print('no_rows')
except Exception as e:
    print(f'parse_fail: {e}')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] OK    $name $best (${elapsed}s)" | tee -a "$MASTER"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $name (rc=$rc, ${elapsed}s)" | tee -a "$MASTER"
        tail -3 "$LOG/$name.err" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
    fi
}

# ==========================================================================
# Stage 8a: small-scale smoke at full pipeline (500 images, 10 epochs)
# Time: ~3-5 min. Validates the full VOC + mAP path.
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 8a: smoke 500 imgs / 10 epochs ===" | tee -a "$MASTER"
TIMEOUT_S=1800 run "voc2007_smoke_n500_e10_s0" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_voc \
        --year 2007 --image-set train --n-images 500 \
        --epochs 10 --input-size 128 --max-objects 6 \
        --batch-size 16 --seed 0 \
        --jsonl-out "$LOG/voc2007_smoke_n500_e10_s0.jsonl"

# ==========================================================================
# Stage 8b: full VOC2007 train, single seed
# 5011 images, 30 epochs, 5 configs. ~60-90 min.
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 8b: full VOC2007 train s=0 ===" | tee -a "$MASTER"
TIMEOUT_S=10800 run "voc2007_train_full_e30_s0" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_voc \
        --year 2007 --image-set train \
        --epochs 30 --input-size 128 --max-objects 8 \
        --batch-size 16 --seed 0 \
        --jsonl-out "$LOG/voc2007_train_full_e30_s0.jsonl"

# ==========================================================================
# Stage 8c: VOC2007 trainval + test mAP (proper held-out detection)
# Train on trainval (5011), evaluate on test (4952).
# Currently train_voc only trains; eval-on-different-split needs
# extension. For now, single training run + skip-validation flagged.
# ==========================================================================
# NOTE: this stage is a placeholder. To do proper VOC mAP evaluation,
# train_voc.py needs a --eval-image-set flag that runs detection
# metrics on a different split after training. Not yet implemented.
# Skip until then.

# ==========================================================================
# Stage 8d: +ricci-mod 5-seed paired (only +ricci-mod and baseline,
# to keep wall time manageable; n_images=2000 to fit 5-seed budget)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 8d: 5-seed +ricci-mod n=2000 ===" | tee -a "$MASTER"
for seed in 0 1 2 3 4; do
    TIMEOUT_S=3600 run "voc2007_n2000_e30_s${seed}" \
        env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -u -m signedkan_wip.src.vision.train_voc \
            --year 2007 --image-set train --n-images 2000 \
            --epochs 30 --input-size 128 --max-objects 8 \
            --batch-size 16 --seed "$seed" \
            --jsonl-out "$LOG/voc2007_n2000_e30_s${seed}.jsonl"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-8 VOC END ===" | tee -a "$MASTER"
# Final summary across all VOC runs
python3 - <<'PY' | tee -a "$MASTER"
import json, pathlib
log_dir = pathlib.Path("reports/overnight_2026_05_11_stage8")
print("\n=== VOC mAP summary ===")
print(f"{'run':<35s}  {'config':<18s}  {'mAP@0.5':>7s}  {'mAP@0.5:95':>10s}  {'mIoU':>5s}")
print("-" * 95)
for f in sorted(log_dir.glob("voc*.jsonl")):
    rows = []
    for line in f.read_text().splitlines():
        try: rows.append(json.loads(line))
        except Exception: pass
    for r in rows:
        m50 = r.get("mAP_50")
        m95 = r.get("mAP_50_95")
        miou = r.get("mean_iou_matched")
        if m50 is None:
            continue
        print(f"{f.stem:<35s}  {r['label']:<18s}  {m50:>7.3f}  {m95:>10.3f}  {miou:>5.3f}")
PY
