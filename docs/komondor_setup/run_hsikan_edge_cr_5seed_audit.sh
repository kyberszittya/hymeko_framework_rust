#!/usr/bin/env bash
#
# HSiKAN-edge_cr 5-seed audit on Slashdot + Epinions (real + shuffle).
# Submitted after the 2026-06-04 hymeko wheel ship + probe 13885703
# validated the full pipeline (Slashdot s=0 = 0.9099, 11 min wall,
# 2 GB MaxRSS — matches published 5-seed SOTA 0.9067 ± .0029).

#SBATCH --job-name=hsikan-edge-cr-audit
#SBATCH --output=slurm_logs/hsikan-edge-cr-audit-%j.out
#SBATCH --error=slurm_logs/hsikan-edge-cr-audit-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pr_szevis

set -uo pipefail
mkdir -p slurm_logs hsikan_edge_cr_audit
REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SIF="$REPO/hymeko_signedkan.sif"
cd "$REPO"

module purge
module load singularity

JSONL=hsikan_edge_cr_audit/results.jsonl
ORCH=hsikan_edge_cr_audit/chain.log
mkdir -p hsikan_edge_cr_audit
> "$JSONL"

ts() { date -Iseconds; }
log() { echo "[edge-cr-audit] $(ts) $*" | tee -a "$ORCH"; }

log "===== HSiKAN-edge_cr 5-seed audit started (slurm job: ${SLURM_JOB_ID:-unknown}) ====="

DATASETS=(slashdot epinions)
SEEDS=(0 1 2 3 4)
HIDDEN=4
N_EPOCHS=80
MAX_K=200000

run_one() {
    local dataset="$1"; local seed="$2"; local mode="$3"
    local extra="" tag="real"
    if [ "$mode" = "shuffle" ]; then
        extra="--shuffle-train-signs"
        tag="shuffle"
    fi
    local cell_log="hsikan_edge_cr_audit/${dataset}_${tag}_seed${seed}.log"
    log "  RUN $dataset seed=$seed mode=$tag"
    local t0; t0=$(date +%s)
    singularity exec --nv --bind "$REPO:/workspace" \
        --env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        --env HSIKAN_ATTENTION_M_E=quaternion \
        --env HSIKAN_ATTENTION_HIGHWAY=1 \
        --env HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        --env HSIKAN_CYCLE_BATCH=2000 \
        --env HSIKAN_MAX_K3=$MAX_K \
        --env HSIKAN_MAX_K2=$MAX_K \
        --env HYMEKO_CYCLE_CACHE=1 \
        --env HYMEKO_CYCLE_CACHE_DIR=/workspace/.cache/hymeko_cycles \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --env MALLOC_ARENA_MAX=4 \
        --env OMP_NUM_THREADS=4 \
        "$SIF" bash -c "cd /workspace && PYTHONPATH=. python -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset $dataset --hidden $HIDDEN --seed $seed \
            --n-epochs $N_EPOCHS --max-k4 $MAX_K $extra" \
        > "$cell_log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local jline=$(grep -E '^\{"dataset"' "$cell_log" | tail -1)
    if [ -n "$jline" ]; then
        echo "$jline" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
d['_audit_mode'] = '$tag'; d['_audit_seed'] = $seed; d['_audit_elapsed_s'] = $elapsed
d['_audit_config'] = 'edge_cr_sota'; d['_audit_stamp'] = '$(date +%Y%m%dT%H%M%SZ)'
print(json.dumps(d))
" >> "$JSONL"
        local auc=$(echo "$jline" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('auc', 'NA'))")
        log "    OK rc=$rc elapsed=${elapsed}s auc=$auc"
    else
        log "    FAIL rc=$rc elapsed=${elapsed}s"
        tail -3 "$cell_log" | sed 's/^/      /' >> "$ORCH"
    fi
}

for dataset in "${DATASETS[@]}"; do
    log "DATASET: $dataset"
    for seed in "${SEEDS[@]}"; do
        for mode in real shuffle; do
            run_one "$dataset" "$seed" "$mode"
        done
    done
done

log "===== HSiKAN-edge_cr 5-seed audit chain complete ====="
log "JSONL: $JSONL ($(wc -l < "$JSONL") rows)"

python3 -c "
import json
from collections import defaultdict
r = defaultdict(list)
with open('$JSONL') as f:
    for line in f:
        d = json.loads(line)
        if isinstance(d.get('auc'), (int, float)):
            r[(d['dataset'], d['_audit_mode'])].append(d['auc'])
print()
for (ds, mode), vals in sorted(r.items()):
    m = sum(vals)/len(vals)
    s = (sum((v-m)**2 for v in vals)/max(len(vals)-1,1))**0.5 if len(vals)>=2 else 0.0
    print(f'HSiKAN-edge_cr  {ds:<14} {mode:<8} n={len(vals)}  auc={m:.4f} ± {s:.4f}')
" 2>&1 | tee -a "$ORCH"
