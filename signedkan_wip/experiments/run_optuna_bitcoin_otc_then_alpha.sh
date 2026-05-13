#!/usr/bin/env bash
# Back-compat wrapper: OTC then Alpha (see ``run_optuna_serial_datasets.sh``).
# For OTC + Alpha + Slashdot + Epinions, use ``run_optuna_core_signed_graphs_serial.sh``.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export OPTUNA_DATASETS="${OPTUNA_DATASETS:-bitcoin_otc bitcoin_alpha}"
exec bash "${DIR}/run_optuna_serial_datasets.sh"
