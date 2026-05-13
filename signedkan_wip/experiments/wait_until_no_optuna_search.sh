#!/usr/bin/env bash
# Wait until no ``python -m signedkan_wip.src.run_optuna_search`` jobs are
# running, then optionally run a follow-up command.  Use this so **new**
# Optuna processes load the current tree (e.g. ``_attention_kind_candidates``
# VRAM gate) after long studies started before a code change.
#
# Does **not** stop running jobs — it only blocks.
#
# Usage:
#   ./signedkan_wip/experiments/wait_until_no_optuna_search.sh
#   nohup ./signedkan_wip/experiments/wait_until_no_optuna_search.sh -- \\
#     bash signedkan_wip/experiments/run_optuna_core_signed_graphs_serial.sh \\
#     >> experiments/results/wait_then_core.log 2>&1 &
#
# Wall-clock first: use ``at(1)`` or ``systemd-run --on-calendar=…`` with a
# one-liner that ``cd`` s to the repo (``at`` does not inherit your cwd).
#
# Env:
#   WAIT_OPTUNA_POLL_S   poll interval (default 60)
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
POLL="${WAIT_OPTUNA_POLL_S:-60}"
# Match Optuna driver module (subprocess ``run_final_cell`` does not).
PAT="signedkan_wip.src.run_optuna_search"

wait_loop() {
  while pgrep -f "${PAT}" >/dev/null 2>&1; do
    echo "[wait_optuna] run_optuna_search still running ($(date -Is)); sleep ${POLL}s" >&2
    sleep "${POLL}"
  done
  echo "[wait_optuna] no matching processes ($(date -Is))" >&2
}

wait_loop

if [[ "${1:-}" == "--" ]]; then
  shift
  if (($# > 0)); then
    echo "[wait_optuna] exec: $*" >&2
    exec "$@"
  fi
fi

echo "[wait_optuna] done — start new Optuna from this tree to pick up latest run_optuna_search.py"
