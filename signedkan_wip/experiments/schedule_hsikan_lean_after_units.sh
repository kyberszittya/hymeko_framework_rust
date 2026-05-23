#!/usr/bin/env bash
# Wait for one or more *user* systemd units to finish, then run the Bitcoin
# HSiKAN lean sweep (see run_hsikan_lean_bitcoin.sh).
#
# Usage (wait for several units in order, then default Bitcoin sweep):
#   ./signedkan_wip/experiments/schedule_hsikan_lean_after_units.sh \
#       run-a.service run-b.service
#
# Or set HSIKAN_LEAN_AFTER_CMD to run a witness / merge instead of the sweep.
#
# Optional env:
#   WAIT_POLL_INTERVAL_S   poll interval (default 30)
#   HSIKAN_LEAN_DEVICE     forwarded when running the Bitcoin sweep (default: cpu)
#   HSIKAN_LEAN_OUT        output JSONL path (default from inner script)
#   HSIKAN_LEAN_AFTER_CMD  if set: ``exec bash -lc "$HSIKAN_LEAN_AFTER_CMD"`` after
#                          all units finish (skips the Bitcoin sweep).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

INTERVAL="${WAIT_POLL_INTERVAL_S:-30}"

if (($# < 1)); then
  echo "usage: $0 <systemd-user-unit.service> [...]" >&2
  echo "example: $0 run-rddecb84693354dcc8fed6ba5efd0fc40.service" >&2
  exit 1
fi

for unit in "$@"; do
  [[ "${unit}" == "--" ]] && continue
  echo "[schedule_hsikan_lean_after_units] waiting until not active: ${unit}"
  while systemctl --user is-active --quiet "${unit}"; do
    systemctl --user show -p ActiveState,SubState "${unit}" 2>/dev/null | tr '\n' ' ' || true
    echo " (poll ${INTERVAL}s)"
    sleep "${INTERVAL}"
  done
  echo "[schedule_hsikan_lean_after_units] ${unit} is no longer active"
done

if [[ -n "${HSIKAN_LEAN_AFTER_CMD:-}" ]]; then
  echo "[schedule_hsikan_lean_after_units] running HSIKAN_LEAN_AFTER_CMD"
  exec bash -lc "${HSIKAN_LEAN_AFTER_CMD}"
fi

export HSIKAN_LEAN_DEVICE="${HSIKAN_LEAN_DEVICE:-cpu}"
echo "[schedule_hsikan_lean_after_units] starting lean sweep (HSIKAN_LEAN_DEVICE=${HSIKAN_LEAN_DEVICE})"
exec bash "${ROOT}/signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh"
