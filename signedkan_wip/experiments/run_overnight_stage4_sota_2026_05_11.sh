#!/bin/bash
# Stage-4 backup: SOTA-breaking on Epinions if FPN alone doesn't
# break the 0.84 ceiling (-0.10 gap vs SGT ~0.95). Composes the
# levers we know work elsewhere:
#   * walks (c3,c4,w2,w3 mix) — broke Slashdot SOTA in
#     project_attention_cycle_batch_compose_2026_05_08.md
#   * wider hidden (h=32 instead of 16) — capacity ceiling test
#   * HSiKAN attention — broke Slashdot at h=4
#   * CPG-3 (concentric pyramid; the moderate ladder) + global ABB
#
# Launches AFTER the main overnight v3 queue completes (or by hand on
# wake if FPN underperformed). Runs sequentially, 16 GB cgroup cap.
#
# Memory caveat: walks (k=2 windows) double cycle count; combined
# with CPG hubs at cap=256 keep memory under 16 GB on Epinions.
# Test with c3+c4 only (no walks) first, then add walks if stable.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage4
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-4 SOTA-break START ===" > "$MASTER"

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
# Stage-1 RETRY (cache fingerprint fixed): Epinions CPG sweep
# ==========================================================================
# The 2026-05-11 silent-cache-correctness bug had _topk_fingerprint not
# including HSIKAN_TOPK_TIERS, HSIKAN_USE_PER_VERTEX_ABB*,
# HSIKAN_VERTEX_FILTER*. All four FPN variants in the primary queue's
# Stage 1 read the same cached cycle set and returned identical AUC.
# Stage 4 retries those configs with the corrected fingerprint.
# Results land in the same Stage-4 dir (epinions_fpn{3,5,7}_g10_s0 +
# noabb), separate from the primary-queue's bogus run names.

TIMEOUT_S=7200 run "epinions_cpg5_g10_s0_RETRY" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:16,100.0:0 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

TIMEOUT_S=7200 run "epinions_cpg3_g10_s0_RETRY" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

TIMEOUT_S=7200 run "epinions_cpg7_g10_s0_RETRY" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,0.5:512,1.0:256,5.0:128,20.0:64,50.0:32,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

TIMEOUT_S=7200 run "epinions_cpg5_noabb_s0_RETRY" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:16,100.0:0 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# CPG-floor variants: keep leaves at cap=16 (preserve leaf signal).
# CPG-5 had bottom 80% at cap=0 — too aggressive. CPG-5-floor16
# keeps the pyramid shape but bottoms out at 16 instead of 0.
TIMEOUT_S=7200 run "epinions_cpg5_floor16_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:32,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# Bitcoin OTC CPG-floor smoke (the original CPG hurt -2.2pp on Bitcoin
# because cap-0 leaves; floor16 may recover).
TIMEOUT_S=1800 run "bitcoin_otc_cpg5_floor16_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=0.1:1024,1.0:256,5.0:64,20.0:32,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset bitcoin_otc --seed 0 --n-epochs 80 \
        --model HSiKAN --hidden 16

# ==========================================================================
# Stage-4: SOTA-breaking levers (kitchen-sink combos)
# ==========================================================================

# Lever 1: wider hidden (h=32) — pure capacity
TIMEOUT_S=7200 run "epinions_h32_off_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# Lever 2: CPG-3 (moderate ladder, leaves cap 16) + g10 ABB + h=32
TIMEOUT_S=7200 run "epinions_h32_cpg3_g10_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:256,10.0:64,100.0:16 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# Lever 3: walks (c3,c4,w2,w3) — what broke Slashdot SOTA
TIMEOUT_S=7200 run "epinions_walks_h16_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# Lever 4: walks + CPG-3 + g10 ABB + h=32 (kitchen sink within memory)
TIMEOUT_S=9000 run "epinions_kitchen_sink_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:128,10.0:32,100.0:8 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.src.run_final_cell \
        --dataset epinions --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-4 SOTA-break END ===" | tee -a "$MASTER"
for f in "$LOG"/*.json; do
    name=$(basename "$f" .json)
    auc=$(tail -1 "$f" 2>/dev/null | python3 -c "
import sys, json
try: print(f'{json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('---')
" 2>/dev/null)
    echo "  $name = $auc" | tee -a "$MASTER"
done
