#!/usr/bin/env bash
# Preset: Optuna ``run_optuna_search`` on the four core signed-graph benchmarks,
# strictly one dataset after another (same SQLite storage, shared stamp).
#
# Default order (override with OPTUNA_DATASETS):
#   bitcoin_otc → bitcoin_alpha → slashdot → epinions
#
# Same env as ``run_optuna_serial_datasets.sh`` (N_TRIALS, N_EPOCHS,
# OPTUNA_STORAGE, OPTUNA_PAUSE_BETWEEN_S, OPTUNA_QUEUE_REST_AT, …).
#
# Overnight handoff when another job is still running: queue this script with
# ``run_after_experiment.sh`` (blocks on the repo CUDA flock + optional PID).
#
# Examples:
#   N_TRIALS=15 ./signedkan_wip/experiments/run_optuna_core_signed_graphs_serial.sh
#   OPTUNA_QUEUE_REST_AT="tomorrow 02:00" ./.../run_optuna_core_signed_graphs_serial.sh
#     # runs OTC now; queues Alpha+Slashdot+Epinions for 02:00 (see serial driver)
#
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export OPTUNA_DATASETS="${OPTUNA_DATASETS:-bitcoin_otc bitcoin_alpha slashdot epinions}"
exec bash "${DIR}/run_optuna_serial_datasets.sh"
