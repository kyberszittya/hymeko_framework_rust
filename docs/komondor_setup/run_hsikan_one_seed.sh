#!/usr/bin/env bash
#
# Run HSiKAN on Komondor. ONE process, ONE seed, ONE GPU, NOT an array.
#
# Direct port of one iteration of the project's working local script
# `signedkan_wip/experiments/run_bitcoin_optuna_best_5seed_2026_05_13.sh`.
# That local script uses `run_final_cell` (a single training run with
# fixed hyperparameters that were previously found by Optuna), NOT
# `run_optuna_search` (which would actually invoke Optuna at runtime).
# This SBATCH does the same: HSiKAN training at the trial-23 best
# config, no in-process Optuna search.
#
# `--cpus-per-task=4` matches the rough core-count visibility of the
# user's local workstation environment where the same workload runs
# at ~1.8 GB RSS without OOM. The container's `OMP_NUM_THREADS=4`
# (baked into %environment) reinforces this from the library side.
#
# Use:
#   sbatch docs/komondor_setup/run_hsikan_one_seed.sh
# Inspect after completion:
#   sacct -j <jobid> -o JobID,State,MaxRSS,Elapsed

#SBATCH --job-name=hsikan-1seed
#SBATCH --output=slurm_logs/hsikan-1seed-%j.out
#SBATCH --error=slurm_logs/hsikan-1seed-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

echo "=== HSiKAN one-seed run start $(date -Iseconds) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Image:   $SIF"
echo ""

# Env vars VERBATIM from the working local script's run_cell call for
# the bitcoin_alpha trial-23 best config. Nothing added, nothing
# changed. `run_final_cell` is a single-trial HSiKAN training; no
# Optuna search invoked.
t0=$(date +%s)
singularity exec --nv --bind "$REPO:/workspace" \
    --env HSIKAN_MIXED_TUPLES=c2,c5,w2,w3,w4 \
    --env HSIKAN_MAX_K3=100000 \
    --env HSIKAN_MAX_K2=100000 \
    --env HSIKAN_ALPHA_ENTROPY_LAMBDA=0.09660950681178301 \
    --env HYMEKO_CYCLE_CACHE=1 \
    --env HSIKAN_CYCLE_BATCH=2000 \
    "$SIF" bash -c "cd /workspace && PYTHONPATH=. python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_alpha --hidden 8 --seed 0 --n-epochs 80 --max-k4 100000"
rc=$?
t1=$(date +%s)

echo ""
echo "=== HSiKAN one-seed run end rc=$rc wall=$((t1-t0))s ==="
exit $rc
