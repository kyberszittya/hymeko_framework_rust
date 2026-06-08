#!/usr/bin/env bash
#
# Local per-dataset wall smoke at the HSiKAN-edge_cr SOTA config.
# Validates the estimator's calibration vs actual local measurement.
#
# Skips datasets that OOM locally (Slashdot + Epinions at SOTA need
# > 7.6 GB GPU). BA + OTC fit.
#
# Usage:
#   bash scripts/local_wall_smoke.sh                # default datasets
#   DATASETS="bitcoin_alpha" bash scripts/local_wall_smoke.sh
#
# Output: per-dataset wall + ratio vs estimator prediction.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

DATASETS="${DATASETS:-bitcoin_alpha bitcoin_otc slashdot epinions}"
PY="${PY:-/home/kyberszittya/miniconda3/bin/python}"
LOG_DIR="/tmp/local_wall_smoke_$(date +%s)"
mkdir -p "$LOG_DIR"
RESULTS="$LOG_DIR/results.tsv"
echo -e "dataset\twall_s\test_cold_s\test_warm_s\tratio_to_cold\tratio_to_warm\tauc" > "$RESULTS"

printf "%-15s %-8s %-12s %-12s %-9s %-9s %s\n" \
    "dataset" "wall_s" "est_cold_s" "est_warm_s" "vs_cold" "vs_warm" "auc"
printf -- "─────────────────────────────────────────────────────────────────────────────\n"

for ds in $DATASETS; do
    # Estimator predictions for both cache states.
    est_cold_hms=$("$PY" scripts/estimate_slurm_time.py --cells "${ds}:real:cold" --no-eff-check 2>/dev/null)
    est_warm_hms=$("$PY" scripts/estimate_slurm_time.py --cells "${ds}:real:warm" --no-eff-check 2>/dev/null)
    # HH:MM:SS -> seconds
    est_cold=$(echo "$est_cold_hms" | awk -F: '{print $1*3600 + $2*60 + $3}')
    est_warm=$(echo "$est_warm_hms" | awk -F: '{print $1*3600 + $2*60 + $3}')

    # Cold-cache run: use a fresh per-smoke cache dir so we measure cold.
    log="$LOG_DIR/${ds}.log"
    cache="$LOG_DIR/cache_${ds}"
    t0=$(date +%s)
    # Production-equivalent env that fits on RTX 2070 SUPER 7.6 GB:
    #   per_vertex K=128 + Triton fused fwd/bwd kernels.
    # The 2026-05-09 Triton backward kernel saves 92-93% of the SOTA
    # forward+backward memory (memory: project_fused_backward_kernel_
    # 2026_05_09); without it Slashdot/Epinions OOM at edge_cr
    # forward. With it: BA ~50s, OTC ~50s, Slashdot ~22min, Epinions
    # fits with bigger_caps (memory: project_epinions_ceiling_2026_05_09).
    env \
        HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_TOPK_MODE=per_vertex \
        HSIKAN_TOPK_K=128 \
        HSIKAN_MAX_K2=200000 \
        HSIKAN_MAX_K3=200000 \
        HSIKAN_TOPK_PRUNER=balance \
        HSIKAN_TOPK_SCORER=fraction_negative \
        HSIKAN_TRITON_KERNEL=1 \
        HSIKAN_TRITON_BACKWARD=1 \
        HYMEKO_CYCLE_CACHE=1 \
        HYMEKO_CYCLE_CACHE_DIR="$cache" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        MALLOC_ARENA_MAX=4 \
        OMP_NUM_THREADS=4 \
        PYTHONPATH=. "$PY" -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset "$ds" --hidden 4 --n-epochs 80 \
            --max-k4 200000 --seed 0 \
            > "$log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))

    auc="NA"
    jline=$(grep -E '^\{"dataset"' "$log" | tail -1)
    [ -n "$jline" ] && auc=$(echo "$jline" | "$PY" -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')

    ratio_cold=$(echo "scale=3; $elapsed / $est_cold" | bc)
    ratio_warm=$(echo "scale=3; $elapsed / $est_warm" | bc)
    printf "%-15s %-8s %-12s %-12s %-9s %-9s %s\n" \
        "$ds" "$elapsed" "$est_cold ($est_cold_hms)" "$est_warm ($est_warm_hms)" \
        "$ratio_cold" "$ratio_warm" "$auc"
    echo -e "${ds}\t${elapsed}\t${est_cold}\t${est_warm}\t${ratio_cold}\t${ratio_warm}\t${auc}" >> "$RESULTS"
    if [ $rc -ne 0 ]; then
        echo "  [error] rc=$rc; tail of log:"; tail -5 "$log" | sed 's/^/    /'
    fi
done

echo
echo "log dir: $LOG_DIR"
echo "results: $RESULTS"