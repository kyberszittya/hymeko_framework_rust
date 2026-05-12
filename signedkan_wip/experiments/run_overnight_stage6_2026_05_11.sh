#!/bin/bash
# Stage 6 (2026-05-11): CPG-continuous architecture — degree-adaptive
# m_v with floor.  Distinguishes from Stage 5's step-tiered CPG.
#
# Hypothesis: the "correct" CPG architecture is a CONTINUOUS function
# of degree, not a step ladder.  The function
#     m_v(deg) = clamp(c · deg, m_min, m_max)
# already exists in `enumerate_top_k_per_vertex_cycles_signed_adaptive_rs`
# (per_vertex_adaptive mode).  Prior single-seed Epinions abbreviated
# showed c=1 lifted AUC +6.7pp vs uniform-cap (memory
# project_degree_adaptive_mv_smoke_2026_05_10.md).  Stage 6 tests this
# on the standard (non-abbreviated) Epinions config + composes with the
# walks lever from Stage 4/5.
#
# This is the architectural counterpart to Stage 5's step-tiered CPG
# and is run in parallel to give two distinct CPG flavours for
# comparison.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage6
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-6 CPG-continuous START ===" > "$MASTER"
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
# Part A: Adaptive m_v sweep on Epinions, h=16 (architecture-only signal)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Part A: adaptive m_v sweep h=16 ===" | tee -a "$MASTER"

# Variant 1: m_v = clamp(1.0 · deg, 32, 512) — replicates morning's +6.7pp config
TIMEOUT_S=3600 run "epinions_adaptive_c1_m32_512_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=512 \
        HSIKAN_TOPK_M_V_MIN=32 \
        HSIKAN_TOPK_M_V_MAX=512 \
        HSIKAN_TOPK_M_V_C=1.0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# Variant 2: m_v = clamp(0.5 · deg, 64, 256) — gentler slope, higher floor
TIMEOUT_S=3600 run "epinions_adaptive_c05_m64_256_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=256 \
        HSIKAN_TOPK_M_V_MIN=64 \
        HSIKAN_TOPK_M_V_MAX=256 \
        HSIKAN_TOPK_M_V_C=0.5 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# Variant 3: m_v = clamp(2.0 · deg, 16, 1024) — aggressive slope, low floor
TIMEOUT_S=3600 run "epinions_adaptive_c2_m16_1024_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=1024 \
        HSIKAN_TOPK_M_V_MIN=16 \
        HSIKAN_TOPK_M_V_MAX=1024 \
        HSIKAN_TOPK_M_V_C=2.0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# Variant 4 (control): uniform floor only, no degree scaling
# m_v = clamp(0.0 · deg, 64, 64) = 64 for all (equivalent to per_vertex K=64)
TIMEOUT_S=3600 run "epinions_adaptive_c0_uniform64_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=64 \
        HSIKAN_TOPK_M_V_MIN=64 \
        HSIKAN_TOPK_M_V_MAX=64 \
        HSIKAN_TOPK_M_V_C=0.0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# ==========================================================================
# Part B: Best adaptive variant × walks × h=32 (compose with Stage 4 winners)
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Part B: adaptive + walks + h=32 ===" | tee -a "$MASTER"

# c=1 + walks + h=32 (continuous CPG kitchen-sink)
TIMEOUT_S=5400 run "epinions_adaptive_c1_walks_h32_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=512 \
        HSIKAN_TOPK_M_V_MIN=32 \
        HSIKAN_TOPK_M_V_MAX=512 \
        HSIKAN_TOPK_M_V_C=1.0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# c=0.5 + walks + h=32 (gentler slope variant)
TIMEOUT_S=5400 run "epinions_adaptive_c05_walks_h32_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=256 \
        HSIKAN_TOPK_M_V_MIN=64 \
        HSIKAN_TOPK_M_V_MAX=256 \
        HSIKAN_TOPK_M_V_C=0.5 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# ==========================================================================
# Part C: Bitcoin OTC validation of best adaptive variant
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === Part C: Bitcoin OTC adaptive sanity ===" | tee -a "$MASTER"

TIMEOUT_S=1800 run "bitcoin_otc_adaptive_c1_m32_512_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=512 \
        HSIKAN_TOPK_M_V_MIN=32 \
        HSIKAN_TOPK_M_V_MAX=512 \
        HSIKAN_TOPK_M_V_C=1.0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-6 END ===" | tee -a "$MASTER"
for f in "$LOG"/*.json; do
    name=$(basename "$f" .json)
    auc=$(tail -1 "$f" 2>/dev/null | python3 -c "
import sys, json
try: print(f'{json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('---')
" 2>/dev/null)
    echo "  $name = $auc" | tee -a "$MASTER"
done
