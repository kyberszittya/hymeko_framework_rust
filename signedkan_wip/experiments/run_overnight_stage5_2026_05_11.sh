#!/bin/bash
# Stage 5 (2026-05-11): 5-seed validation of kitchen-sink + CPG
# architecture tuning. Sequenced so the 5-seed paired result lands
# even if the architecture exploration runs long.
#
# Hypothesis under test for CPG:
#   The original CPG ladder (e.g. CPG-5 = 0.1:1024,...,100.0:0) zeros
#   out the bottom 80% of vertices. In HSiKAN, every vertex needs SOME
#   cycles to populate its row of M_e (the cycle-incidence matrix). A
#   zero-cap leaf becomes an all-zero M_e row → that vertex has no
#   learning signal → predictions on it fail → AUC drops.
#   The fix: SOFT PYRAMIDS — no zero tier, every vertex keeps a floor
#   cap so its M_e row stays populated.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage5
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-5 START ===" > "$MASTER"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] git SHA $(git rev-parse HEAD)" >> "$MASTER"
fi

run() {
    local name="$1"; shift
    if [ -s "$LOG/$name.json" ] && grep -q '"auc"' "$LOG/$name.json" 2>/dev/null; then
        local auc
        auc=$(tail -1 "$LOG/$name.json" | python3 -c "
import sys, json
try: print(f'AUC={json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('reparse_fail')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] SKIP  $name $auc" | tee -a "$MASTER"
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
        local auc
        auc=$(tail -1 "$LOG/$name.json" 2>/dev/null | python3 -c "
import sys, json
try: print(f'AUC={json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('no_auc')
" 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] OK    $name $auc (${elapsed}s)" | tee -a "$MASTER"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $name (rc=$rc, ${elapsed}s)" | tee -a "$MASTER"
        tail -3 "$LOG/$name.err" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
    fi
}

# ==========================================================================
# Part A: 5-seed paired validation of kitchen-sink (Stage 4 winner 0.8141)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Part A: 5-seed kitchen-sink paired ===" | tee -a "$MASTER"

for seed in 0 1 2 3 4; do
    # Baseline: m=64 OFF h=16 (no walks, no CPG)
    TIMEOUT_S=3600 run "epinions_baseline_s${seed}" \
        env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
            HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            HYMEKO_CYCLE_CACHE=1 \
        python -u -m signedkan_wip.src.run_final_cell \
            --dataset epinions --seed "$seed" --n-epochs 60 \
            --model HSiKAN --hidden 16
    # Kitchen-sink: walks + CPG-3 + g10 + h=32
    TIMEOUT_S=5400 run "epinions_kitchen_sink_s${seed}" \
        env HSIKAN_TOPK_MODE=per_vertex_tiered \
            HSIKAN_TOPK_TIERS=1.0:128,10.0:32,100.0:8 \
            HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
            HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
            HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
            HSIKAN_TORCH_COMPILE=0 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            HYMEKO_CYCLE_CACHE=1 \
        python -u -m signedkan_wip.src.run_final_cell \
            --dataset epinions --seed "$seed" --n-epochs 60 \
            --model HSiKAN --hidden 32
done

# ==========================================================================
# Part B: CPG architecture tuning — soft pyramids (NO zero tier)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Part B: CPG soft pyramids ===" | tee -a "$MASTER"

# Hypothesis: zero-tier kills M_e rows. Soft pyramid keeps every
# vertex at >=floor cycles. Test multiple gradients.

# CPG-soft-gentle: 8× range, floor 32. Hubs get 256, leaves get 32.
TIMEOUT_S=3600 run "epinions_cpg_soft_gentle_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:32 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# CPG-soft-steep: 16× range, floor 64. Hubs get 1024, leaves get 64.
TIMEOUT_S=3600 run "epinions_cpg_soft_steep_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:512,5.0:256,20.0:128,100.0:64 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# CPG-uniform-128 (control: same total budget as soft-gentle, uniform)
TIMEOUT_S=3600 run "epinions_uniform128_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=128 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# CPG-soft-gentle + walks + h=32 (apply lessons together)
TIMEOUT_S=5400 run "epinions_cpg_soft_gentle_walks_h32_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:32 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# Walks h=32 (no CPG) — isolates walks+h32 from kitchen-sink CPG-3.
TIMEOUT_S=5400 run "epinions_walks_h32_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# ==========================================================================
# Part C: Bitcoin OTC CPG-soft validation (CPG-original hurt -2.2pp;
#         does CPG-soft recover by keeping leaf signal?)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Part C: Bitcoin OTC CPG-soft ===" | tee -a "$MASTER"

TIMEOUT_S=1800 run "bitcoin_otc_cpg_soft_gentle_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:32 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

TIMEOUT_S=1800 run "bitcoin_otc_cpg_soft_steep_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:512,5.0:256,20.0:128,100.0:64 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

# ==========================================================================
# Summary
# ==========================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-5 END ===" | tee -a "$MASTER"
for f in "$LOG"/*.json; do
    name=$(basename "$f" .json)
    auc=$(tail -1 "$f" 2>/dev/null | python3 -c "
import sys, json
try: print(f'{json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('---')
" 2>/dev/null)
    echo "  $name = $auc" | tee -a "$MASTER"
done
