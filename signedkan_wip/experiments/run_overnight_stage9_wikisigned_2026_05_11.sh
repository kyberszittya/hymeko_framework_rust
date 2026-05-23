#!/bin/bash
# Stage 9 (2026-05-11): WikiSigned — third signed-graph benchmark.
#
# Goal: another shot at SOTA. WikiSigned (Maniu et al.) is 138K nodes,
# 740K edges, ~70% positive — denser than Slashdot, less hub-heavy
# than Epinions.  If local cycle/walk primitives close the SGT gap
# here (where they failed on Epinions), it's a real second-SOTA
# data point for the paper.
#
# Pipeline:
#   1. download + load smoke (5 min)
#   2. baseline (m=64 OFF h=16, 60 epochs)
#   3. kitchen-sink (walks c3,c4,w2,w3 + CPG-3 + g10 ABB + h=32, 60 epochs)
#   4. edge_cr config (Slashdot SOTA winner) + seed 0
#
# Per CLAUDE.md §4: 16 GB cgroup cap per run.

set -uo pipefail
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust

LOG=reports/overnight_2026_05_11_stage9
mkdir -p "$LOG"
MASTER="$LOG/MASTER.log"
if [ ! -s "$MASTER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-9 WikiSigned START ===" > "$MASTER"
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
    local timeout_s="${TIMEOUT_S:-7200}"
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

# Trigger download via Python
echo "[$(date '+%H:%M:%S')] pre-fetch wikisigned dataset" | tee -a "$MASTER"
python3 -c "from signedkan_wip.src.datasets import load; g = load('wikisigned'); print(f'|V|={g.n_nodes} |E|={len(g.edges)}')" 2>&1 | tee -a "$MASTER"

# Baseline
TIMEOUT_S=3600 run "wikisigned_baseline_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wikisigned --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

# Kitchen-sink (proven Epinions +7.53pp)
TIMEOUT_S=5400 run "wikisigned_kitchen_sink_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:128,10.0:32,100.0:8 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wikisigned --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# edge_cr (Slashdot SOTA recipe)
TIMEOUT_S=7200 run "wikisigned_edge_cr_s0" \
    env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_MAX_K3=200000 \
        HSIKAN_MAX_K2=200000 \
        HSIKAN_TRITON_KERNEL=1 \
        HSIKAN_TRITON_BACKWARD=1 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wikisigned --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 0 --model HSiKAN

# ==========================================================================
# wiki_elec: Wikipedia admin elections, 7,118 nodes / 103,675 edges
# Small + dense votes — should be FAST and a good local-structure test.
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === wiki_elec: baseline + kitchen-sink + edge_cr ===" \
    | tee -a "$MASTER"
TIMEOUT_S=1800 run "wiki_elec_baseline_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wiki_elec --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

TIMEOUT_S=3600 run "wiki_elec_kitchen_sink_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:128,10.0:32,100.0:8 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wiki_elec --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

TIMEOUT_S=5400 run "wiki_elec_edge_cr_s0" \
    env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_MAX_K3=200000 \
        HSIKAN_MAX_K2=200000 \
        HSIKAN_TRITON_KERNEL=1 \
        HSIKAN_TRITON_BACKWARD=1 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wiki_elec --hidden 4 --n-epochs 80 \
        --max-k4 100000 --seed 0 --model HSiKAN

# ==========================================================================
# wiki_conflict: Wikipedia edit-war conflict network
# 116K nodes / 2M edges (post-binarize-dedup). Biggest dataset we run.
# Use smaller k4 cap to keep memory bounded.
# ==========================================================================
echo "[$(date '+%H:%M:%S')] === wiki_conflict: baseline + kitchen-sink + edge_cr ===" \
    | tee -a "$MASTER"
TIMEOUT_S=3600 run "wiki_conflict_baseline_s0" \
    env HSIKAN_TOPK_MODE=per_vertex HSIKAN_TOPK_K=64 \
        HSIKAN_USE_PER_VERTEX_ABB=0 HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wiki_conflict --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 16

TIMEOUT_S=5400 run "wiki_conflict_kitchen_sink_s0" \
    env HSIKAN_TOPK_MODE=per_vertex_tiered \
        HSIKAN_TOPK_TIERS=1.0:128,10.0:32,100.0:8 \
        HSIKAN_USE_PER_VERTEX_ABB=1 HSIKAN_USE_PER_VERTEX_ABB_MODE=global \
        HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE=1.0 \
        HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wiki_conflict --seed 0 --n-epochs 60 \
        --model HSiKAN --hidden 32

# edge_cr — smaller caps for wiki_conflict due to 2M edges
TIMEOUT_S=10800 run "wiki_conflict_edge_cr_s0" \
    env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_MAX_K3=100000 \
        HSIKAN_MAX_K2=100000 \
        HSIKAN_TRITON_KERNEL=1 \
        HSIKAN_TRITON_BACKWARD=1 \
        HSIKAN_TORCH_COMPILE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HYMEKO_CYCLE_CACHE=1 \
    python -u -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset wiki_conflict --hidden 4 --n-epochs 60 \
        --max-k4 50000 --seed 0 --model HSiKAN

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage-9 WikiSigned END ===" | tee -a "$MASTER"
for f in "$LOG"/*.json; do
    name=$(basename "$f" .json)
    auc=$(tail -1 "$f" 2>/dev/null | python3 -c "
import sys, json
try: print(f'{json.loads(sys.stdin.read())[\"auc\"]:.4f}')
except: print('---')
" 2>/dev/null)
    echo "  $name = $auc" | tee -a "$MASTER"
done
