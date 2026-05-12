#!/usr/bin/env bash
# Bitcoin Alpha/OTC: HSiKAN lean sweep (enumerator + ABB + SSG profile).
# From repo root.  JSONL default under reports/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

# Strict UV policy: always resolve interpreter from uv workspace env.
if ! command -v uv >/dev/null 2>&1; then
  echo "[run_hsikan_lean_bitcoin] uv not found; install uv" >&2
  exit 2
fi
PYTHON="${ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[run_hsikan_lean_bitcoin] missing ${PYTHON}; run: uv sync --group ml --all-packages" >&2
  exit 3
fi
if ! "${PYTHON}" -c "import torch, numpy" >/dev/null 2>&1; then
  echo "[run_hsikan_lean_bitcoin] ${PYTHON} missing torch/numpy; run: uv sync --group ml --all-packages" >&2
  exit 4
fi
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${HSIKAN_LEAN_OUT:-reports/hsikan_lean_bitcoin_${STAMP}.jsonl}"

LEAN_DEV_FLAGS=()
if [[ -n "${HSIKAN_LEAN_DEVICE:-}" ]]; then
  LEAN_DEV_FLAGS=( --device "${HSIKAN_LEAN_DEVICE}" )
fi

"${PYTHON}" -m signedkan_wip.src.run_hsikan_lean_profile \
  --datasets bitcoin_alpha bitcoin_otc \
  --seeds 0 1 2 \
  --hidden 8 12 16 \
  --profiles clean_baseline pv_k128 pv_k128_abb_g10 pv_k64_abb_g10 pv_k64_abb_ssg_deg3 \
  --n-epochs 80 \
  --timeout-s 7200 \
  --python "${PYTHON}" \
  "${LEAN_DEV_FLAGS[@]}" \
  --out "${OUT}"

echo "[run_hsikan_lean_bitcoin] wrote ${OUT}"
