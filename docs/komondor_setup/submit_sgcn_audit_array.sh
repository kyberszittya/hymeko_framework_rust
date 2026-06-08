#!/usr/bin/env bash
#
# SGCN label-shuffle audit — Komondor job array.
#
# Grid: 4 datasets × 2 modes × 3 seeds = 24 cells.
#   datasets: bitcoin_alpha, bitcoin_otc, slashdot, epinions
#   modes:    real (no flag), shuffle (--shuffle-train-signs)
#   seeds:    0, 1, 2
#
# Reddit not yet in run_final_cell.py's dispatch; deferred to a
# follow-up after a small dataset-loader wire-up.
#
# Each cell is one SLURM array task. Each task ~30s-10min wall depending
# on dataset size (Bitcoin small; Slashdot/Epinions larger). Parallel
# A100 throughput puts the whole grid under ~20-30 min total.
#
# Results: one JSON line per task in slurm_logs/sgcn-audit-<arr>.out;
# the trailing reduce-step (sgcn_audit_summary.sh, separate) aggregates.

#SBATCH --job-name=sgcn-audit
#SBATCH --output=slurm_logs/sgcn-audit-%A_%a.out
#SBATCH --error=slurm_logs/sgcn-audit-%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

# ─── Grid decomposition ────────────────────────────────────────
# Order of variation (slowest-changing first): dataset, mode, seed.
# So: idx = dataset_idx * 6 + mode_idx * 3 + seed_idx
DATASETS=(bitcoin_alpha bitcoin_otc slashdot epinions)
MODES=(real shuffle)
SEEDS=(0 1 2)

idx=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( idx % 3 ))
mode_idx=$(( (idx / 3) % 2 ))
ds_idx=$(( idx / 6 ))

DATASET=${DATASETS[$ds_idx]}
MODE=${MODES[$mode_idx]}
SEED=${SEEDS[$seed_idx]}

EXTRA_FLAGS=""
if [ "$MODE" = "shuffle" ]; then
    EXTRA_FLAGS="--shuffle-train-signs"
fi

# ─── Provenance header ─────────────────────────────────────────
echo "=== ARRAY $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID start: $(date -Iseconds) ==="
echo "Grid:         dataset=$DATASET mode=$MODE seed=$SEED"
echo "Node:         ${SLURM_NODELIST:-?} (partition=${SLURM_JOB_PARTITION:-?})"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Singularity:  $(singularity --version)"
echo "Image:        $SIF"
echo ""

# ─── The run ────────────────────────────────────────────────────
t0=$(date +%s)
singularity exec --nv --bind "$REPO:/workspace" "$SIF" \
    bash -c "cd /workspace && PYTHONPATH=. python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset $DATASET --model SGCN --hidden 32 --seed $SEED --n-epochs 120 \
        $EXTRA_FLAGS"
rc=$?
t1=$(date +%s)

echo ""
echo "=== ARRAY $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID end: rc=$rc wall=$((t1-t0))s ==="
exit $rc
