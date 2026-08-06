#!/usr/bin/env bash
#
# Single-seed Komondor probe for HSiKAN-edge_cr config — first run after
# the 2026-06-04 hymeko wheel ship. Verifies the Rust extension is
# actually picked up inside the container (so cycle / walk enumeration
# stays bounded), and reproduces the published 5-seed Slashdot SOTA
# (0.9067 ± .0029 per reports/2026-05-09-slashdot-edge-cr-sota.md)
# at n=1.

#SBATCH --job-name=hsikan-edge-cr-probe
#SBATCH --output=slurm_logs/hsikan-edge-cr-probe-%j.out
#SBATCH --error=slurm_logs/hsikan-edge-cr-probe-%j.err
#SBATCH --time=00:45:00
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

echo "=== HSiKAN-edge_cr probe start $(date -Iseconds) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Image: $SIF (built 2026-06-04 with hymeko wheel)"
echo ""
echo "=== verifying hymeko Rust extension inside container ==="
singularity exec "$SIF" python -c \
    "import hymeko; print('hymeko ok:', hymeko.__file__); \
     import inspect; print('walks_rs:', inspect.signature(hymeko.enumerate_top_k_walks_rs))"
echo ""

# HSiKAN-edge_cr published SOTA config (Slashdot 5-seed 0.9067 ± .0029)
t0=$(date +%s)
singularity exec --nv --bind "$REPO:/workspace" \
    --env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
    --env HSIKAN_ATTENTION_M_E=quaternion \
    --env HSIKAN_ATTENTION_HIGHWAY=1 \
    --env HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
    --env HSIKAN_CYCLE_BATCH=2000 \
    --env HSIKAN_MAX_K3=200000 \
    --env HSIKAN_MAX_K2=200000 \
    --env HYMEKO_CYCLE_CACHE=1 \
    --env HYMEKO_CYCLE_CACHE_DIR=/workspace/.cache/hymeko_cycles \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env MALLOC_ARENA_MAX=4 \
    --env OMP_NUM_THREADS=4 \
    "$SIF" bash -c "cd /workspace && PYTHONPATH=. python -m hymeko_neuro.experiments.runs.run_final_cell \
        --dataset slashdot --hidden 4 --seed 0 --n-epochs 80 --max-k4 200000"
rc=$?
t1=$(date +%s)

echo ""
echo "=== HSiKAN-edge_cr probe end rc=$rc wall=$((t1-t0))s ==="
exit $rc
