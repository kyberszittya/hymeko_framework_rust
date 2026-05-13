# shellcheck shell=bash
# Source-only. Activates the uv-managed venv + loads project env vars.
#
# Usage:
#   . scripts/dev/activate.sh
#
# Composes:
#   1. .venv/bin/activate  (uv-generated; sets PATH + VIRTUAL_ENV)
#   2. .env                (project-specific HSIKAN_/HYMEKO_/determinism vars)
#
# `.env` is optional. Copy `.env.example` to `.env` and edit to opt in.
# Equivalent one-shot: `uv run --env-file .env <cmd>`.

if [ -n "${BASH_SOURCE-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "scripts/dev/activate.sh must be sourced, not executed." >&2
    exit 1
fi

_hymeko_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/../.." && pwd)"

if [ ! -f "${_hymeko_repo_root}/.venv/bin/activate" ]; then
    echo "No .venv at ${_hymeko_repo_root}/.venv — run \`uv sync --group ml --group demo\` first." >&2
    return 1
fi

# shellcheck disable=SC1091
. "${_hymeko_repo_root}/.venv/bin/activate"

if [ -f "${_hymeko_repo_root}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${_hymeko_repo_root}/.env"
    set +a
    echo "[hymeko] activated .venv + loaded .env"
else
    echo "[hymeko] activated .venv (no .env present — copy .env.example to opt in)"
fi

unset _hymeko_repo_root
