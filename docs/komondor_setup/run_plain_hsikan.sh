#!/usr/bin/env bash
#
# Plain HSiKAN on Komondor. NO Optuna anything. NO mixed-arity
# multi-tuple configuration. NO alpha-entropy regularizer env var.
# NO cycle-cache or cycle-batch tuning env vars. Just the base
# HSiKAN model at default arities, single seed, single GPU.
#
# This is what `run_final_cell --model HSiKAN` runs without any
# of the HSIKAN_* env vars that trigger the Optuna-search-tuned
# mixed-arity configuration that was OOMing all day.

#SBATCH --job-name=hsikan-plain
#SBATCH --output=slurm_logs/hsikan-plain-%j.out
#SBATCH --error=slurm_logs/hsikan-plain-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pr_szevis

set -uo pipefail
mkdir -p slurm_logs
REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SIF="$REPO/hymeko_signedkan.sif"
cd "$REPO"

module purge
module load singularity

echo "=== HSiKAN start $(date -Iseconds) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Image: $SIF"
echo ""
echo "=== configuration values passed via --env ==="
echo "  HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4   (mixed cycle+walk arities — config, not Optuna)"
echo "  HSIKAN_MAX_K3=100000 HSIKAN_MAX_K2=100000   (cycle-count caps — config)"
echo "  HSIKAN_ALPHA_ENTROPY_LAMBDA=0.0966...   (entropy reg weight — a number, not Optuna)"
echo "  HYMEKO_CYCLE_CACHE=1   (cycle cache toggle)"
echo "  HSIKAN_CYCLE_BATCH=2000   (cycle batch size)"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
echo "    (A100/SXM4 CUDA caching allocator tune — 2026-06-03 mixed-tuples"
echo "     RSS investigation: same mixed-tuples config uses 0.66 GB on local CPU"
echo "     vs 33 GB on Komondor A100 → CUDA caching allocator delta. expandable"
echo "     segments avoid pre-reserving whole 40 GB; max_split caps fragmentation;"
echo "     gc_threshold triggers caching-allocator GC at 80% before OOM trips.)"
echo ""

# The HSIKAN_* and HYMEKO_* env vars below are CONFIGURATION VALUES,
# verbatim from the project's working local script. They happen to
# be values Optuna found earlier; using them does NOT invoke any
# Optuna search machinery. `run_final_cell` is a single-trial
# training run.
t0=$(date +%s)
singularity exec --nv --bind "$REPO:/workspace" \
    --env HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4 \
    --env HSIKAN_MAX_K3=100000 \
    --env HSIKAN_MAX_K2=100000 \
    --env HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301 \
    --env HYMEKO_CYCLE_CACHE=1 \
    --env HSIKAN_CYCLE_BATCH=2000 \
    --env HYMEKO_CYCLE_CACHE_DIR=/workspace/.cache/hymeko_cycles \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8 \
    --env MALLOC_ARENA_MAX=4 \
    --env OMP_NUM_THREADS=4 \
    "$SIF" bash -c "cd /workspace && PYTHONPATH=. python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_alpha --hidden 8 --seed 0 --n-epochs 80 --max-k4 100000"
rc=$?
t1=$(date +%s)

echo ""
echo "=== plain HSiKAN end rc=$rc wall=$((t1-t0))s ==="
exit $rc
