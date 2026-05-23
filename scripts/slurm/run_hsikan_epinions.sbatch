#!/usr/bin/env bash
#
# Komondor / SLURM submission script — run HSiKAN on Epinions at the
# published recipe.  Local 8 GB GPU OOMs at h=16 with full arities;
# Komondor's A100 (40 GB) or H100 (80 GB) clears the M_vt /
# spline-forward working set comfortably.
#
# Edit the SBATCH directives below with your account / partition.
# Then submit with:
#   sbatch scripts/slurm/run_hsikan_epinions.sbatch
#
# ── REQUIRED user-supplied fields ──────────────────────────────────
#SBATCH --job-name=hsikan-epinions
#SBATCH --account=YOUR_ACCOUNT             # <-- e.g. 'sztaki' or your project ID
#SBATCH --partition=gpu                     # <-- gpu / gpu-bw / gpu-h100 etc.
#SBATCH --gres=gpu:1                        # one GPU is plenty
#SBATCH --cpus-per-task=8                   # cycle enumeration is rayon-parallel
#SBATCH --mem=64G                           # M_vt / M_e construction wants RAM
#SBATCH --time=02:00:00                     # ~3min/seed × 3 seeds + slack
#SBATCH --output=logs/hsikan_epinions_%j.log
#SBATCH --error=logs/hsikan_epinions_%j.err

set -euo pipefail
mkdir -p logs

# ── conda env (or module load) ─────────────────────────────────────
# If Komondor exposes a shared pytorch module:
#   module load pytorch/2.1
# Otherwise activate the user's miniconda environment:
#   source $HOME/miniconda3/etc/profile.d/conda.sh
#   conda activate hsikan
# Adjust to whatever Komondor's pattern is — left as a TODO.

# ── repo root ──────────────────────────────────────────────────────
REPO=${REPO:-$HOME/hymeko_framework_rust}
cd "$REPO"

# ── ensure the pyo3 wheel is built for the local arch ──────────────
# If hymeko_py wheel is already built into the active env, this is
# a no-op:
if ! python3 -c "import hymeko" 2>/dev/null; then
  cd hymeko_py && maturin develop --release && cd ..
fi

# ── 3-seed Epinions sweep ──────────────────────────────────────────
OUT=signedkan_wip/experiments/results/hsikan_epinions_komondor.jsonl
mkdir -p "$(dirname "$OUT")"
> "$OUT"

for SEED in 0 1 2; do
  echo "=== seed=$SEED $(date) ===" | tee -a "${OUT%.jsonl}.log"
  HSIKAN_TORCH_COMPILE=1 \
  HSIKAN_ARITIES=2,3,4 \
  HSIKAN_CHUNK_T=0 \
  python3 -m signedkan_wip.src.run_final_cell \
    --dataset epinions --model HSiKAN \
    --hidden 16 --n-epochs 100 --max-k4 200000 \
    --seed "$SEED" \
    | tee -a "${OUT%.jsonl}.log" \
    | tail -1 \
    >> "$OUT"
done

echo "=== done $(date) ==="
echo "Results: $OUT"
echo "Aggregate:"
python3 - "$OUT" <<'PY'
import json, sys, statistics
aucs = []; f1s = []
for l in open(sys.argv[1]):
    if not l.strip() or l.strip() == "null":
        continue
    r = json.loads(l)
    if "auc" in r:
        aucs.append(r["auc"]); f1s.append(r["f1m"])
if aucs:
    print(f"  AUC = {statistics.mean(aucs):.4f} ± "
          f"{statistics.stdev(aucs):.4f}  (n={len(aucs)})")
    print(f"  F1m = {statistics.mean(f1s):.4f} ± "
          f"{statistics.stdev(f1s):.4f}")
PY
