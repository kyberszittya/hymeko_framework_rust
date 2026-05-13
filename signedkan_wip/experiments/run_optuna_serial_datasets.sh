#!/usr/bin/env bash
# Sequential Optuna: run ``run_optuna_search`` on each dataset in
# ``OPTUNA_DATASETS`` (space-separated), one after another, same storage URL.
#
# Preset for OTC + Alpha + Slashdot + Epinions:
#   ``run_optuna_core_signed_graphs_serial.sh`` (same env / ``at`` options).
#
# Optional **defer** of all datasets *after* the first: set
# ``OPTUNA_QUEUE_REST_AT`` to an ``at(1)`` timespec (e.g. ``now + 2 hours``,
# ``tomorrow 06:00``).  The remainder runs in a new ``at`` job so the GPU is
# free until then.  Requires ``at`` installed and permission to use it.
#
# Env (common):
#   OPTUNA_DATASETS   default: bitcoin_otc bitcoin_alpha (override or use preset script)
#   OPTUNA_STAMP      optional; fixed study suffix (default: UTC stamp once)
#   N_TRIALS, N_EPOCHS, OPTUNA_STORAGE, HSIKAN_CYCLE_BATCH — same as bitcoin wrapper
#   OPTUNA_PAUSE_BETWEEN_S  sleep seconds between back-to-back datasets (default 0)
#   OPTUNA_QUEUE_REST_AT    if set, after dataset[0] finishes, ``at`` the rest
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO"
export HSIKAN_CYCLE_BATCH="${HSIKAN_CYCLE_BATCH:-2000}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("error: CUDA required for this Optuna driver", file=sys.stderr)
    sys.exit(2)
PY

N_TRIALS="${N_TRIALS:-30}"
N_EPOCHS="${N_EPOCHS:-80}"
STAMP="${OPTUNA_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OPTUNA_STAMP="$STAMP"

if [[ -z "${OPTUNA_STORAGE:-}" ]]; then
  export OPTUNA_STORAGE="sqlite:///signedkan_wip/experiments/results/optuna_serial_${STAMP}.db"
fi

read -r -a DATASETS <<<"${OPTUNA_DATASETS:-bitcoin_otc bitcoin_alpha}"
if ((${#DATASETS[@]} == 0)); then
  echo "[optuna_serial] error: empty OPTUNA_DATASETS" >&2
  exit 1
fi

echo "[optuna_serial] storage=${OPTUNA_STORAGE}"
echo "[optuna_serial] stamp=${STAMP}  datasets(${#DATASETS[@]}): ${DATASETS[*]}"

run_one() {
  local ds=$1
  echo "[optuna_serial] start dataset=${ds} trials=${N_TRIALS} epochs=${N_EPOCHS}"
  python -m signedkan_wip.src.run_optuna_search \
    --dataset "${ds}" \
    --n-trials "${N_TRIALS}" \
    --n-epochs "${N_EPOCHS}" \
    --study-name "${ds}_${STAMP}" \
    --storage "${OPTUNA_STORAGE}" \
    --sampler "${OPTUNA_SAMPLER:-tpe}"
  echo "[optuna_serial] done dataset=${ds}"
}

run_one "${DATASETS[0]}"

REST=("${DATASETS[@]:1}")
if ((${#REST[@]} == 0)); then
  echo "[optuna_serial] all datasets finished storage=${OPTUNA_STORAGE}"
  exit 0
fi

JOINED="${REST[*]}"

if [[ -n "${OPTUNA_QUEUE_REST_AT:-}" ]]; then
  if ! command -v at >/dev/null 2>&1; then
    echo "[optuna_serial] error: OPTUNA_QUEUE_REST_AT set but \`at\` not found" >&2
    exit 2
  fi
  JOBDIR="${REPO}/signedkan_wip/experiments/results"
  mkdir -p "${JOBDIR}"
  JOBFILE="${JOBDIR}/.optuna_at_${STAMP}_$$.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(printf %q "${REPO}")"
    echo "export PYTHONPATH=$(printf %q "${REPO}")"
    echo "export HSIKAN_CYCLE_BATCH=$(printf %q "${HSIKAN_CYCLE_BATCH}")"
    echo "export OPTUNA_STORAGE=$(printf %q "${OPTUNA_STORAGE}")"
    echo "export OPTUNA_STAMP=$(printf %q "${STAMP}")"
    echo "export N_TRIALS=$(printf %q "${N_TRIALS}")"
    echo "export N_EPOCHS=$(printf %q "${N_EPOCHS}")"
    echo "export OPTUNA_SAMPLER=$(printf %q "${OPTUNA_SAMPLER:-tpe}")"
    echo "export OPTUNA_DATASETS=$(printf %q "${JOINED}")"
    echo "unset OPTUNA_QUEUE_REST_AT"
    echo "exec bash $(printf %q "${REPO}/signedkan_wip/experiments/run_optuna_serial_datasets.sh")"
  } >"${JOBFILE}"
  chmod +x "${JOBFILE}"
  # ``at`` reads the job from stdin on most Linux installs.
  at "${OPTUNA_QUEUE_REST_AT}" <"${JOBFILE}"
  rm -f "${JOBFILE}"
  echo "[optuna_serial] queued ${#REST[@]} dataset(s) at ${OPTUNA_QUEUE_REST_AT}: ${JOINED}"
  echo "[optuna_serial] check: atq   remove: atrm <jobid>"
  exit 0
fi

for ds in "${REST[@]}"; do
  sleep "${OPTUNA_PAUSE_BETWEEN_S:-0}"
  run_one "${ds}"
done

echo "[optuna_serial] all datasets finished storage=${OPTUNA_STORAGE}"
