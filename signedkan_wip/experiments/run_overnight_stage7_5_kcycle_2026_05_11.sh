#!/bin/bash
# Stage 7.5 (2026-05-11): KCycle micro-graph HyMeYOLO at full scale,
# direct comparator to +ricci-mod from Stage 7 Round 1.
#
# Single config (+kcycle), single seed, n=10K × 100 epochs, matching
# the Stage 7 hymeyolo_ricci_n10k_e100_s0 baseline scale so the +kcycle
# result is directly comparable mAP-for-mAP.
#
# After this, the 5-seed paired KCycle vs +ricci-mod runs in
# Stage 7's existing s1..s4 (which now include +kcycle as config 6 of
# the train_circles_ricci ablation set).
#
# Per CLAUDE.md §4: 16 GB cgroup cap per run.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage7_5
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-7.5 KCycle START ===" > "$MASTER"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] git SHA $(git rev-parse HEAD)" >> "$MASTER"
fi

run() {
    local name="$1"; shift
    if [ -s "$LOG/$name.jsonl" ] && grep -q '"label"' "$LOG/$name.jsonl" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name" | tee -a "$MASTER"
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
        local metric
        metric=$(python3 -c "
import json
try:
    rows = [json.loads(l) for l in open('$LOG/$name.jsonl')]
    if rows:
        r = rows[-1]
        print(f\"mAP50={r.get('mAP_50',0):.3f} mAP50:95={r.get('mAP_50_95',0):.3f}\")
except Exception:
    print('no_metric')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] OK    $name $metric (${elapsed}s)" | tee -a "$MASTER"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $name (rc=$rc, ${elapsed}s)" | tee -a "$MASTER"
        tail -3 "$LOG/$name.err" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
    fi
}

# Stage 7's Round 1 HyMeYOLO at n=10K×100ep timed out at config 4 of 5,
# so +ricci-mod NEVER ran at that scale. Re-run both +ricci-mod and
# +kcycle here so we get an apples-to-apples comparison at the same
# scale (n=10K, 100 epochs, seed=0). Wall budget for 2 configs at
# ~1700s/config ≈ 3400s. 3h timeout has plenty of headroom.
TIMEOUT_S=10800 run "hymeyolo_ricci_kcycle_n10k_e100_s0" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 10000 --epochs 100 --seed 0 \
        --configs "+ricci-mod,+kcycle" \
        --jsonl-out "$LOG/hymeyolo_ricci_kcycle_n10k_e100_s0.jsonl"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-7.5 KCycle END ===" | tee -a "$MASTER"
