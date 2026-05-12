#!/bin/bash
# Stage 7 (2026-05-11 afternoon): Epinions SOTA push + HyMeYOLO CV
# in parallel via interleaved sequential queue.
#
# Epinions ladder — close the gap to SGT ~0.95 from current 0.8145:
#   A. Reproduce edge_cr 5-seed (prior was 0.8464 mean, +0.032 over
#      kitchen-sink) — warm-cache, ~10-15 min/seed instead of 85 min.
#   B. edge_cr × walks fusion (the kitchen-sink walks recipe combined
#      with the edge_cr highway gate + quaternion attention; novel).
#   C. h=8 attention variant (between h=4 edge_cr and h=32 kitchen-sink).
#
# CV ladder — HyMeYOLO Cluttered MNIST scale-up:
#   D. +ricci-mod at n_images=10000, 100 epochs (4× current scale).
#   E. +ricci-mod 5-seed at n_images=5000, 50 epochs (validation).
#
# Interleaved so neither stream stalls on a long single run.
# Per CLAUDE.md §4: 16 GB cgroup cap per run.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage7
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-7 SOTA push + CV START ===" > "$MASTER"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] git SHA $(git rev-parse HEAD)" >> "$MASTER"
fi

run() {
    local name="$1"; shift
    if [ -s "$LOG/$name.json" ] && grep -q '"auc"\|"box_cls_acc"\|"label"' "$LOG/$name.json" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] SKIP  $name (already complete)" | tee -a "$MASTER"
        return 0
    fi
    local start=$(date +%s)
    echo "[$(date '+%H:%M:%S')] START $name" | tee -a "$MASTER"
    local timeout_s="${TIMEOUT_S:-5400}"
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope --quiet \
            -p MemoryMax=16G -p MemorySwapMax=0 \
            timeout "$timeout_s" "$@" > "$LOG/$name.json" 2> "$LOG/$name.err"
    else
        timeout "$timeout_s" "$@" > "$LOG/$name.json" 2> "$LOG/$name.err"
    fi
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    if [ $rc -eq 0 ]; then
        local metric
        metric=$(tail -1 "$LOG/$name.json" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    if 'auc' in d: print(f'AUC={d[\"auc\"]:.4f}')
    elif 'box_cls_acc' in d: print(f'box_acc={d[\"box_cls_acc\"]:.2f} circ_acc={d.get(\"circ_cls_acc\", -1):.2f}')
    else: print('done')
except: print('parse_fail')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] OK    $name $metric (${elapsed}s)" | tee -a "$MASTER"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $name (rc=$rc, ${elapsed}s)" | tee -a "$MASTER"
        tail -3 "$LOG/$name.err" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
    fi
}

# Shared env: edge_cr recipe (Slashdot SOTA winner)
ENV_EDGE_CR=(
    HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3
    HSIKAN_ATTENTION_M_E=quaternion
    HSIKAN_ATTENTION_HIGHWAY=1
    HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr
    HSIKAN_CYCLE_BATCH=2000
    HSIKAN_MAX_K3=200000
    HSIKAN_MAX_K2=200000
    HSIKAN_TRITON_KERNEL=1
    HSIKAN_TRITON_BACKWARD=1
    HSIKAN_TORCH_COMPILE=0
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    HYMEKO_CYCLE_CACHE=1
)

# ==========================================================================
# Round 1: Epinions edge_cr seed=0 reproduction + CV scale-up
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Round 1: Epinions edge_cr s=0 + CV scale ===" | tee -a "$MASTER"

# A: edge_cr seed 0 (proves the warm-cache speedup; prior was 0.8409 single-seed)
TIMEOUT_S=7200 run "epinions_edge_cr_s0" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 0 --model HSiKAN

# D: HyMeYOLO scale-up n=10000 epochs=100 seed=0 (CV parallel)
TIMEOUT_S=7200 run "hymeyolo_ricci_n10k_e100_s0" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 10000 --epochs 100 --seed 0 \
        --jsonl-out "$LOG/hymeyolo_ricci_n10k_e100_s0.jsonl"

# ==========================================================================
# Round 2: Epinions edge_cr seeds 1-2 + CV 5-seed (first 2)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Round 2: edge_cr seeds 1-2 + HyMeYOLO s0-1 ===" | tee -a "$MASTER"

TIMEOUT_S=7200 run "epinions_edge_cr_s1" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 1 --model HSiKAN

TIMEOUT_S=3600 run "hymeyolo_ricci_n5k_e50_s1" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 5000 --epochs 50 --seed 1 \
        --jsonl-out "$LOG/hymeyolo_ricci_n5k_e50_s1.jsonl"

TIMEOUT_S=7200 run "epinions_edge_cr_s2" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 2 --model HSiKAN

TIMEOUT_S=3600 run "hymeyolo_ricci_n5k_e50_s2" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 5000 --epochs 50 --seed 2 \
        --jsonl-out "$LOG/hymeyolo_ricci_n5k_e50_s2.jsonl"

# ==========================================================================
# Round 3: Epinions edge_cr seeds 3-4 + CV 5-seed (last 2)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Round 3: edge_cr seeds 3-4 + HyMeYOLO s3-4 ===" | tee -a "$MASTER"

TIMEOUT_S=7200 run "epinions_edge_cr_s3" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 3 --model HSiKAN

TIMEOUT_S=3600 run "hymeyolo_ricci_n5k_e50_s3" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 5000 --epochs 50 --seed 3 \
        --jsonl-out "$LOG/hymeyolo_ricci_n5k_e50_s3.jsonl"

TIMEOUT_S=7200 run "epinions_edge_cr_s4" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 4 --model HSiKAN

TIMEOUT_S=3600 run "hymeyolo_ricci_n5k_e50_s4" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.vision.train_circles_ricci \
        --n-images 5000 --epochs 50 --seed 4 \
        --jsonl-out "$LOG/hymeyolo_ricci_n5k_e50_s4.jsonl"

# ==========================================================================
# Round 4: NOVEL Epinions variants (edge_cr × kitchen-sink fusion etc.)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Round 4: novel Epinions variants ===" | tee -a "$MASTER"

# B1: edge_cr at h=8 (between SOTA h=4 and kitchen-sink h=32)
TIMEOUT_S=7200 run "epinions_edge_cr_h8_s0" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 8 --n-epochs 80 \
        --max-k4 100000 --seed 0 --model HSiKAN

# B2: edge_cr at h=16 (kitchen-sink capacity at edge_cr config)
TIMEOUT_S=7200 run "epinions_edge_cr_h16_s0" \
    env "${ENV_EDGE_CR[@]}" \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --hidden 16 --n-epochs 80 \
        --max-k4 100000 --seed 0 --model HSiKAN

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-7 END ===" | tee -a "$MASTER"
for f in "$LOG"/*.json; do
    name=$(basename "$f" .json)
    metric=$(tail -1 "$f" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    if 'auc' in d: print(f'{d[\"auc\"]:.4f}')
    elif 'box_cls_acc' in d: print(f'box={d[\"box_cls_acc\"]:.2f}')
    else: print('---')
except: print('---')
" 2>/dev/null)
    echo "  $name = $metric" | tee -a "$MASTER"
done
