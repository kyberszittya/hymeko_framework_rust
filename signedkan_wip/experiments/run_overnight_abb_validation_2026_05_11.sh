#!/bin/bash
# Overnight ABB-global-fullness + FPN-tiered validation queue (2026-05-11 v3).
#
# Stages (sequential, single GPU at a time, ordered by SOTA-breaking
# importance so that if the queue runs out of time the headline result
# still lands):
#
#   Stage 1. Epinions FPN-tiered × global-min ABB sweep (1 seed each):
#            * FPN-5 ladder + g10 (the headline)
#            * FPN-3 ladder + g10 (moderate)
#            * FPN-7 ladder + g10 (fine-grained)
#            * baseline (fixed m=64, OFF) for comparison
#   Stage 2. Bitcoin Alpha + OTC × 5 seeds × {OFF, global gate=1.0}
#            — pure AUC-preservation validation, paired.
#   Stage 3. Slashdot 1-seed gate=1.0 sanity.
#
# CLAUDE.md §4 mandates a 16 GB memory cap; we enforce via
# `systemd-run --user --scope -p MemoryMax=16G` per-run (cgroup v2 RSS
# cap, kills the run on actual physical-memory overrun rather than on
# virtual-address-space allocation — `ulimit -v` breaks CUDA).
#
# Per CLAUDE.md §11: halts only on diagnosed bugs; transient failures
# are logged and the queue continues so the user has data on wake.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"

# Don't clobber an existing master log if we're resuming.
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Overnight v3 START ===" > "$MASTER"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] git SHA $(git rev-parse HEAD)" >> "$MASTER"
fi

run() {
    local name="$1"; shift
    # Skip if already completed with non-empty JSON.
    if [ -s "$LOG/$name.json" ] && grep -q '"auc"' "$LOG/$name.json" 2>/dev/null; then
        local auc
        auc=$(tail -1 "$LOG/$name.json" | python3 -c "
import sys, json
try: print(f'AUC={json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('reparse_fail')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] SKIP  $name $auc (already complete)" | tee -a "$MASTER"
        return 0
    fi
    local start=$(date +%s)
    echo "[$(date '+%H:%M:%S')] START $name" | tee -a "$MASTER"
    local timeout_s="${TIMEOUT_S:-3600}"
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
        local auc
        auc=$(tail -1 "$LOG/$name.json" 2>/dev/null | python3 -c "
import sys, json
try: print(f'AUC={json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('no_auc')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] OK    $name $auc (${elapsed}s)" | tee -a "$MASTER"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $name (rc=$rc, ${elapsed}s, tail of err:)" | tee -a "$MASTER"
        tail -3 "$LOG/$name.err" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
    fi
}

# ==========================================================================
# Stage 0: Bitcoin OTC FPN smoke (validate FPN+ABB combo before Epinions)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 0: Bitcoin OTC FPN smoke ===" | tee -a "$MASTER"

# Each FPN variant on Bitcoin OTC seed=0 with global ABB gate=1.0.
# Baseline (fixed m=128) already covered by Stage 2's bitcoin_otc_g10_s0.
TIMEOUT_S=1800 run "bitcoin_otc_fpn5_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:16,100.0:0 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

TIMEOUT_S=1800 run "bitcoin_otc_fpn3_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

TIMEOUT_S=1800 run "bitcoin_otc_fpn7_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,0.5:512,1.0:256,5.0:128,20.0:64,50.0:32,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

# Tiered-only (no ABB) on Bitcoin OTC — isolates FPN contribution
TIMEOUT_S=1800 run "bitcoin_otc_fpn5_noabb_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:16,100.0:0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

# ==========================================================================
# Stage 1: Epinions FPN-tiered × global-min ABB (SOTA-breaking headline)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 1: Epinions FPN-tiered + global-min ABB ===" \
    | tee -a "$MASTER"

# Stage-1 baseline: fixed-m=64, ABB OFF (the prior overnight ceiling)
TIMEOUT_S=5400 run "epinions_off_m64_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# FPN-5 ladder (most aggressive: top 0.1% hubs get cap 1024, bottom 80% get 0)
TIMEOUT_S=5400 run "epinions_fpn5_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:16,100.0:0 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# FPN-3 ladder (moderate)
TIMEOUT_S=5400 run "epinions_fpn3_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# FPN-7 ladder (fine-grained, leaves still keep cap 16)
TIMEOUT_S=5400 run "epinions_fpn7_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,0.5:512,1.0:256,5.0:128,20.0:64,50.0:32,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# FPN-5 + tiered-only (no ABB) — isolates the FPN contribution
TIMEOUT_S=5400 run "epinions_fpn5_noabb_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:16,100.0:0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# ==========================================================================
# Stage 2: Bitcoin Alpha + OTC, 5-seed paired
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 2: Bitcoin 5-seed paired ===" | tee -a "$MASTER"

ENV_OFF=(
    HSIKAN_TOPK_MODE=per_vertex
    HSIKAN_TOPK_K=128
    HSIKAN_USE_PER_VERTEX_ABB=0
    HSIKAN_TORCH_COMPILE=0
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
ENV_G10=(
    HSIKAN_TOPK_MODE=per_vertex
    HSIKAN_TOPK_K=128
    HSIKAN_USE_PER_VERTEX_ABB=1
    HSIKAN_USE_PER_VERTEX_ABB_MODE=global
    HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0
    HSIKAN_TORCH_COMPILE=0
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

for seed in 0 1 2 3 4; do
    for ds in bitcoin_alpha bitcoin_otc; do
        TIMEOUT_S=1800 run "${ds}_off_s${seed}" \
            env "${ENV_OFF[@]}" \
            python -u -m signedkan_wip.experiments.runs.run_final_cell \
                --dataset "$ds" --seed "$seed" --n-epochs 80 \
                --model HSiKAN --hidden 16
        TIMEOUT_S=1800 run "${ds}_g10_s${seed}" \
            env "${ENV_G10[@]}" \
            python -u -m signedkan_wip.experiments.runs.run_final_cell \
                --dataset "$ds" --seed "$seed" --n-epochs 80 \
                --model HSiKAN --hidden 16
    done
done

# ==========================================================================
# Stage 3: Slashdot single-seed gate=1.0
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Stage 3: Slashdot single seed gate=1.0 ===" \
    | tee -a "$MASTER"
TIMEOUT_S=3600 run "slashdot_g10_s0" \
    env "${ENV_G10[@]}" \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset slashdot --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16 --max-k4 200000

# ==========================================================================
# Summary
# ==========================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Overnight v3 END ===" | tee -a "$MASTER"
echo "" | tee -a "$MASTER"
echo "Results dump:" | tee -a "$MASTER"
for f in "$LOG"/*.json; do
    name=$(basename "$f" .json)
    auc=$(tail -1 "$f" 2>/dev/null | python3 -c "
import sys, json
try: print(f'{json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('---')
" 2>/dev/null)
    echo "  $name = $auc" | tee -a "$MASTER"
done
