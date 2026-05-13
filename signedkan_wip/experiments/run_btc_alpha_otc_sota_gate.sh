#!/usr/bin/env bash
# Multi-seed HSiKAN gate: bitcoin_alpha + bitcoin_otc vs sota_reference.json
# (self-mode: mean ≥ hsikan_mean − 2·hsikan_std when std known).
#
# **CUDA only** — exits 1 if no GPU (never falls back to CPU).
#
# Serialization: ``run_hsikan_sota_gate`` acquires the repo-wide CUDA flock
# (``signedkan_wip/src/benchmarks/cuda_job_lock.py``) while ``--device cuda``.
# Do not set ``HYMEKO_CUDA_DISABLE_JOB_LOCK`` unless you intentionally overlap
# jobs on one GPU.
#
# Usage:
#   ./signedkan_wip/experiments/run_btc_alpha_otc_sota_gate.sh
#
# Logs: signedkan_wip/experiments/results/btc_alpha_otc_gate_<UTC>.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# Chunked forward (same default spirit as Optuna / overnight scripts).
# For Bitcoin, run_final_cell otherwise leaves cycle_batch_size unset →
# higher peak VRAM. If you still OOM (e.g. sparse M_e build), try lowering
# caps: --max-k4 100000 --max-k3 15000, and keep the GPU free of other jobs.
export HSIKAN_CYCLE_BATCH="${HSIKAN_CYCLE_BATCH:-2000}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("error: CUDA is required for btc_alpha_otc_sota_gate (no CPU fallback)", file=sys.stderr)
    sys.exit(1)
PY

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="signedkan_wip/experiments/results/btc_alpha_otc_gate_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"

python -m signedkan_wip.src.benchmarks.run_hsikan_sota_gate \
  --datasets bitcoin_alpha bitcoin_otc \
  --seeds 0 1 2 3 4 \
  --hidden 16 \
  --n-epochs 80 \
  --max-k4 200000 \
  --device cuda \
  --require-cuda \
  --mode self \
  --k-sigma 2.0 \
  2>&1 | tee "$LOG"
echo "[done] log: $LOG"
