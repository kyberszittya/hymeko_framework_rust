#!/usr/bin/env bash
# Pro scheduler: wait for dependent user units, enforce UV runtime, and
# run HSiKAN lean sweep with device guardrails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

POLL_S="${HSIKAN_PRO_POLL_S:-30}"
WAIT_TIMEOUT_S="${HSIKAN_PRO_WAIT_TIMEOUT_S:-0}"   # 0 = no timeout
LOCKFILE="${HSIKAN_PRO_LOCKFILE:-/tmp/hsikan_lean_pro_scheduler.lock}"
DEVICE="${HSIKAN_PRO_DEVICE:-cpu}"
CUDA_MIN_FREE_MB="${HSIKAN_PRO_CUDA_MIN_FREE_MB:-4096}"
AFTER_CMD="${HSIKAN_PRO_AFTER_CMD:-}"

if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" && "${DEVICE}" != "auto" ]]; then
  echo "[hsikan_pro] invalid HSIKAN_PRO_DEVICE=${DEVICE} (expected cpu|cuda|auto)" >&2
  exit 2
fi

exec 9>"${LOCKFILE}"
if ! flock -n 9; then
  echo "[hsikan_pro] another scheduler is already active: ${LOCKFILE}" >&2
  exit 3
fi

if (($# < 1)); then
  echo "usage: $0 <systemd-user-unit.service> [...]" >&2
  echo "example: $0 run-aaa.service run-bbb.service" >&2
  exit 1
fi

start_ts=$(date +%s)
for unit in "$@"; do
  [[ "${unit}" == "--" ]] && continue
  echo "[hsikan_pro] waiting for ${unit}"
  while systemctl --user is-active --quiet "${unit}"; do
    if (( WAIT_TIMEOUT_S > 0 )); then
      now=$(date +%s)
      if (( now - start_ts > WAIT_TIMEOUT_S )); then
        echo "[hsikan_pro] timeout waiting for ${unit} after ${WAIT_TIMEOUT_S}s" >&2
        exit 4
      fi
    fi
    systemctl --user show -p ActiveState,SubState "${unit}" 2>/dev/null | tr '\n' ' ' || true
    echo " (poll ${POLL_S}s)"
    sleep "${POLL_S}"
  done
  echo "[hsikan_pro] ${unit} not active"
done

# Strict UV policy for scheduler-triggered runs.
if ! command -v uv >/dev/null 2>&1; then
  echo "[hsikan_pro] uv not found on PATH; install uv" >&2
  exit 5
fi
PYTHON="${ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[hsikan_pro] missing ${PYTHON}; run: uv sync --group ml --all-packages" >&2
  exit 6
fi
if ! "${PYTHON}" -c "import torch, numpy" >/dev/null 2>&1; then
  echo "[hsikan_pro] ${PYTHON} missing torch/numpy; run: uv sync --group ml --all-packages" >&2
  exit 7
fi

echo "[hsikan_pro] PYTHON=${PYTHON}"

if [[ "${DEVICE}" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[hsikan_pro] DEVICE=cuda but nvidia-smi not found" >&2
    exit 7
  fi
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ -z "${free_mb}" ]]; then
    echo "[hsikan_pro] unable to read free GPU memory" >&2
    exit 8
  fi
  if (( free_mb < CUDA_MIN_FREE_MB )); then
    echo "[hsikan_pro] refusing cuda run: free=${free_mb}MB < required=${CUDA_MIN_FREE_MB}MB" >&2
    exit 9
  fi
  echo "[hsikan_pro] cuda guard passed: free=${free_mb}MB"
fi

export HSIKAN_LEAN_DEVICE="${DEVICE}"
export PYTHON

if [[ -n "${AFTER_CMD}" ]]; then
  echo "[hsikan_pro] running post-wait command"
  exec bash -lc "${AFTER_CMD}"
fi

echo "[hsikan_pro] starting lean sweep with device=${HSIKAN_LEAN_DEVICE}"
exec bash "${ROOT}/signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh"
