#!/usr/bin/env bash
# Run a command **after** the current CUDA-heavy driver is done — intended for
# overnight handoff (e.g. Optuna on OTC finishes → start the four-graph queue).
#
# 1) Optional: if ``RUN_AFTER_PID`` is set, poll until that process exits (use
#    ``echo $!`` from the shell that started the foreground experiment, or
#    ``pgrep -f run_optuna``).
# 2) Block on the same **flock** file as ``cuda_job_lock.py`` until no driver
#    holds it (matches Optuna / SOTA gate / chase entrypoints).
# 3) ``exec`` your follow-up command (inherits cwd; set ``cd`` inside cmd if needed).
#
# Example before bed (current Optuna already running in another terminal):
#
#   cd /path/to/hymeko_framework_rust
#   nohup env N_TRIALS=30 PYTHONPATH=$PWD \
#     bash signedkan_wip/experiments/run_after_experiment.sh \
#     bash signedkan_wip/experiments/run_optuna_core_signed_graphs_serial.sh \
#     >> signedkan_wip/experiments/results/follow_optuna_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
#
# If the running job is *not* using the repo CUDA flock, set ``RUN_AFTER_PID``
# to its main PID so we wait for exit before step (2).
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK="${HYMEKO_CUDA_JOB_LOCK:-${REPO}/signedkan_wip/experiments/results/.cuda_job_serial.lock}"

if (($# < 1)); then
  echo "usage: RUN_AFTER_PID=<optional> $0 <command> [args...]" >&2
  exit 1
fi

if [[ -n "${RUN_AFTER_PID:-}" ]]; then
  echo "[run_after] waiting for PID ${RUN_AFTER_PID} to exit..." >&2
  while kill -0 "${RUN_AFTER_PID}" 2>/dev/null; do
    sleep "${RUN_AFTER_POLL_S:-30}"
  done
  echo "[run_after] PID ${RUN_AFTER_PID} gone ($(date -Is))" >&2
fi

mkdir -p "$(dirname "${LOCK}")"
exec 200>>"${LOCK}"
echo "[run_after] waiting for CUDA job lock: ${LOCK}" >&2
flock 200
flock -u 200
echo "[run_after] lock free, starting: $* ($(date -Is))" >&2
exec "$@"
