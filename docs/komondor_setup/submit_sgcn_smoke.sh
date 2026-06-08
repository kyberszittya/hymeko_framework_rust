#!/usr/bin/env bash
#
# Komondor probe SLURM job — single SGCN training run on bitcoin_alpha.
# Expected wall < 60 s on an A100 40GB; expected AUC ≈ 0.870
# (matches sgcn_baseline.json seed=0 on the local RTX 2070 SUPER).
#
# Cluster: komondor.hpc.dkf.hu (KIFÜ HUN-REN), per docs.hpc.dkf.hu.
# Path:    Singularity container path (preferred). Venv fallback below.
#
# Usage from /scratch/<PROJECT>/hymeko/hymeko_framework_rust:
#   sbatch docs/komondor_setup/submit_sgcn_smoke.sh

#SBATCH --job-name=sgcn-smoke
#SBATCH --output=slurm_logs/sgcn-smoke-%j.out
#SBATCH --error=slurm_logs/sgcn-smoke-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# ─── Komondor partition / billing (REQUIRED) ─────────────────────────
# `gpu` = 58 nodes × 4× A100 40GB SXM (general GPU work). For DDP that
# benefits from 8 GPUs/node, switch to `ai`.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Account is REQUIRED on Komondor for SLURM accounting / billing.
# (User pr_szhc, project pr_szevis, per sacctmgr 2026-06-02.)
#SBATCH --account=pr_szevis
# QOS options: lowpri (cheaper, deprioritised) | normal (default, used).
# #SBATCH --qos=normal

set -uo pipefail
mkdir -p slurm_logs

# Resolve the repo path. The script is committed inside the repo, so
# `git rev-parse --show-toplevel` works regardless of where it's run.
REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SIF="$REPO/hymeko_signedkan.sif"
cd "$REPO"

# ─── Module loads ────────────────────────────────────────────────────
# Singularity is the canonical execution path. Per Komondor docs, the
# pytorch module already wires CUDA, so we do NOT load CUDA separately.
module purge
module load singularity

# ─── Provenance (committed to log header) ────────────────────────────
echo "=== JOB $SLURM_JOB_ID start: $(date -Iseconds) ==="
echo "Host:         $(hostname)"
echo "Node:         ${SLURM_NODELIST:-?}"
echo "Partition:    ${SLURM_JOB_PARTITION:-?}"
echo "Account:      ${SLURM_JOB_ACCOUNT:-?}"
echo "GPU(s):       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo 'nvidia-smi unavailable')"
echo "Git SHA:      $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
echo "Singularity:  $(singularity --version 2>&1)"
echo "Working dir:  $REPO"
echo ""

# ─── Path selection: container preferred, venv fallback ──────────────
if [ -f "$SIF" ]; then
    EXEC_MODE="singularity"
    echo "Path:         singularity ($SIF)"
    singularity inspect --labels "$SIF" 2>&1 | head -10
    echo ""
    singularity exec --nv --bind "$REPO:/workspace" "$SIF" \
        python -c 'import torch; print("Torch:       ", torch.__version__, "cuda_available=", torch.cuda.is_available())'
elif [ -f "$REPO/.venv/bin/activate" ]; then
    EXEC_MODE="venv"
    echo "Path:         venv (singularity image not built; run Step 2 of README)"
    source "$REPO/.venv/bin/activate"
    echo "Python:       $(python --version)  ($(which python))"
    echo "Torch:        $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())')"
else
    echo "ERROR: no execution environment available."
    echo "Build the image (recommended):"
    echo "  module load singularity"
    echo "  singularity build --fakeroot --fix-perms hymeko_signedkan.sif docs/komondor_setup/hymeko_signedkan.def"
    echo "OR create a venv (fallback) per docs/komondor_setup/README.md Step 2."
    exit 2
fi
echo ""

# ─── The actual probe ────────────────────────────────────────────────
echo ">>> SGCN smoke: bitcoin_alpha, hidden=32, n_epochs=120, seed=0"
echo ""

t0=$(date +%s)
if [ "$EXEC_MODE" = "singularity" ]; then
    singularity exec --nv --bind "$REPO:/workspace" "$SIF" \
        bash -c "cd /workspace && PYTHONPATH=. python -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset bitcoin_alpha --model SGCN --hidden 32 --seed 0 --n-epochs 120"
    rc=$?
else
    export PYTHONPATH=.
    python -m signedkan_wip.experiments.runs.run_final_cell \
        --dataset bitcoin_alpha --model SGCN --hidden 32 --seed 0 --n-epochs 120
    rc=$?
fi
t1=$(date +%s)

echo ""
echo "=== JOB $SLURM_JOB_ID end:   $(date -Iseconds) rc=$rc wall=$((t1-t0))s ==="

# Final-state diagnostics
echo "Peak RSS (this job, KB): $(grep VmHWM /proc/$$/status 2>/dev/null | awk '{print $2}')"
echo "GPU final state:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 || true

exit $rc
