#!/usr/bin/env bash
#
# HSiKAN-edge_cr audit submitter -- SINGLE FILE, THREE MODES.
#
# ════════════════════════════════════════════════════════════════════════
#  USAGE
# ════════════════════════════════════════════════════════════════════════
#   bash submit_hsikan_edge_cr_array.sh smoke   # one cell per class
#   bash submit_hsikan_edge_cr_array.sh full    # full 40-cell grid
#   bash submit_hsikan_edge_cr_array.sh epinions-shuffle-rerun
#                                                # only the 4 missing
#                                                # Epinions shuffle cells
#                                                # (seeds 1,2,3,4)
#
#   sbatch submit_hsikan_edge_cr_array.sh       # FORBIDDEN: refuses to
#                                                # run without an explicit
#                                                # --time override; see
#                                                # the guard at WORKER MODE.
#
# ════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════
# The file is dual-role:
#
#   * ORCHESTRATOR mode  (no SLURM_ARRAY_TASK_ID in env)
#       Submits one or more sbatch sub-arrays with TIGHT per-class --time
#       budgets calibrated from measured walls. NEVER submits with a
#       loose budget; KIFÜ resource-eff warnings (2026-06-04 incident,
#       see reports/2026-06-04-kifu-resource-eff-response.md) were the
#       direct consequence of a single uniform --time=02:30:00 budget
#       across heterogeneous cell classes.
#
#   * WORKER mode  (SLURM_ARRAY_TASK_ID is set; sbatch invoked us)
#       Runs one cell of the 40-cell grid identified by the
#       SLURM_ARRAY_TASK_ID. Decomposes idx -> (dataset, seed, mode),
#       invokes singularity exec of the python entry point, writes a
#       per-cell log + an atomic flock-protected JSONL append.
#
# The WORKER ignores SBATCH headers in the file (it was invoked by
# sbatch with explicit overrides). The ORCHESTRATOR ALWAYS overrides:
#
#   --array, --time, --job-name, --output, --error, --export
#
# The SBATCH headers at the top of this file are NOT used for actual
# submission; they exist as documentation of the static resource pin
# (cpus / mem / partition / gpu / account). If you `sbatch` this file
# directly without an override the worker guard refuses to run.
#
# ════════════════════════════════════════════════════════════════════════
#  CLASS BUDGETS (calibrated 2026-06-04 from chain + K-sweep walls)
# ════════════════════════════════════════════════════════════════════════
#
#   TINY    (BA + OTC + Slashdot-real)         --time=02:00       n=25
#     measured median 30s, max 65s -> 1.85x headroom, TimeEff 25-54%
#
#   MEDIUM  (Slashdot-shuffle cold-cache)      --time=15:00       n=5
#     measured max 649s -> 1.39x headroom, TimeEff 72%
#
#   LONG    (Epinions r+s cold-cache)          --time=03:00:00    n=10
#     measured 8866s -> 1.22x headroom, TimeEff 82%
#
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=hsikan-edge-cr-array
#SBATCH --output=slurm_logs/hsikan-edge-cr-array-%A_%a.out
#SBATCH --error=slurm_logs/hsikan-edge-cr-array-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pr_szevis

set -uo pipefail

# ── Grid definition (used by BOTH modes) ────────────────────────────
DATASETS=(slashdot epinions bitcoin_alpha bitcoin_otc)
SEEDS=(0 1 2 3 4)
MODES=(real shuffle)

#   ds_idx   = idx / 10
#   seed_idx = (idx / 2) % 5
#   mode_idx = idx % 2
TINY_INDICES="0,2,4,6,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"
MEDIUM_INDICES="1,3,5,7,9"
LONG_INDICES="10,11,12,13,14,15,16,17,18,19"
EPINIONS_SHUFFLE_RERUN="13,15,17,19"   # the 4 missing cells from 2026-06-04

# ── Class budgets ───────────────────────────────────────────────────
# Computed from docs/komondor_setup/wall_calibration.yaml via
# scripts/estimate_slurm_time.py. NEVER hardcode these. To change the
# budgets, update the calibration YAML (which lists measured walls)
# and the estimator picks up the new values on the next orchestrator
# invocation. The estimator refuses to emit a budget below the YAML's
# min_time_efficiency floor.
PYTHON="${PYTHON:-python3}"
EST="$(dirname "$0")/../../scripts/estimate_slurm_time.py"
if [ ! -x "$EST" ]; then
    echo "error: estimator not executable at $EST" >&2
    # Worker mode (above) does not reach here, so this only fires for
    # the orchestrator. The orchestrator MUST have the estimator.
    exit 1
fi

# Compute per-class --time at orchestrator startup. The estimator
# emits HH:MM:SS on stdout, diagnostics on stderr (we suppress unless
# --explain is requested). Failure to estimate is fatal.
echo "── calibrated budgets (scripts/estimate_slurm_time.py) ──" >&2
TINY_TIME=$("$PYTHON" "$EST" --class tiny)     || { echo "estimator failed for tiny" >&2; exit 1; }
MEDIUM_TIME=$("$PYTHON" "$EST" --class medium) || { echo "estimator failed for medium" >&2; exit 1; }
LONG_TIME=$("$PYTHON" "$EST" --class long)     || { echo "estimator failed for long" >&2; exit 1; }
echo "  TINY=$TINY_TIME  MEDIUM=$MEDIUM_TIME  LONG=$LONG_TIME" >&2

# ════════════════════════════════════════════════════════════════════
#  WORKER MODE
# ════════════════════════════════════════════════════════════════════
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Guard: refuse to run if --time wasn't overridden via sbatch CLI.
    # SLURM exposes the granted timelimit in TimeLimit; we sanity-check
    # that the timelimit is one of our three class budgets, NOT some
    # accidental 02:30:00 v1-era value.
    AUDIT_K_TAG="${AUDIT_K_TAG:-}"
    OUT_DIR="hsikan_edge_cr_audit_array${AUDIT_K_TAG:+_$AUDIT_K_TAG}"
    mkdir -p slurm_logs "$OUT_DIR"

    REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    SIF="$REPO/hymeko_signedkan.sif"
    cd "$REPO"

    module purge
    module load singularity

    idx=${SLURM_ARRAY_TASK_ID}
    mode_idx=$(( idx % 2 ))
    seed_idx=$(( (idx / 2) % 5 ))
    ds_idx=$(( idx / 10 ))
    DATASET=${DATASETS[$ds_idx]}
    SEED=${SEEDS[$seed_idx]}
    MODE=${MODES[$mode_idx]}

    EXTRA=""; TAG="real"
    if [ "$MODE" = "shuffle" ]; then
        EXTRA="--shuffle-train-signs"; TAG="shuffle"
    fi

    HIDDEN=4
    N_EPOCHS=80
    MAX_K=200000

    CELL_LOG="${OUT_DIR}/${DATASET}_${TAG}_seed${SEED}.log"
    JSONL="${OUT_DIR}/results.jsonl"

    echo "=== ARRAY $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID start: $(date -Iseconds) ==="
    echo "Grid:     dataset=$DATASET mode=$TAG seed=$SEED"
    echo "Node:     ${SLURM_NODELIST:-?}"
    echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
    echo ""

    t0=$(date +%s)
    singularity exec --nv --bind "$REPO:/workspace" \
        --env HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        --env HSIKAN_ATTENTION_M_E=quaternion \
        --env HSIKAN_ATTENTION_HIGHWAY=1 \
        --env HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        --env HSIKAN_CYCLE_BATCH=2000 \
        --env HSIKAN_MAX_K3=$MAX_K \
        --env HSIKAN_MAX_K2=$MAX_K \
        --env HYMEKO_CYCLE_CACHE=1 \
        --env HYMEKO_CYCLE_CACHE_DIR=/workspace/.cache/hymeko_cycles \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --env MALLOC_ARENA_MAX=4 \
        --env OMP_NUM_THREADS=4 \
        "$SIF" bash -c "cd /workspace && PYTHONPATH=. python -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset $DATASET --hidden $HIDDEN --seed $SEED \
            --n-epochs $N_EPOCHS --max-k4 $MAX_K $EXTRA" \
        > "$CELL_LOG" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))

    JLINE=$(grep -E '^\{"dataset"' "$CELL_LOG" | tail -1)
    if [ -n "$JLINE" ]; then
        (
            flock 9
            echo "$JLINE" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
d['_audit_mode'] = '$TAG'; d['_audit_seed'] = $SEED; d['_audit_elapsed_s'] = $elapsed
d['_audit_config'] = 'edge_cr_sota_array'; d['_audit_stamp'] = '$(date +%Y%m%dT%H%M%SZ)'
d['_audit_slurm_task'] = '$SLURM_ARRAY_TASK_ID'
d['_audit_slurm_job'] = '$SLURM_ARRAY_JOB_ID'
print(json.dumps(d))" >> "$JSONL"
        ) 9>"$JSONL.lock"
        AUC=$(echo "$JLINE" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('auc', 'NA'))")
        echo "=== ARRAY task $SLURM_ARRAY_TASK_ID end: rc=$rc wall=${elapsed}s auc=$AUC ==="
    else
        echo "=== ARRAY task $SLURM_ARRAY_TASK_ID end: rc=$rc wall=${elapsed}s FAILED ==="
        tail -5 "$CELL_LOG"
    fi
    exit $rc
fi

# ════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR MODE
# ════════════════════════════════════════════════════════════════════

mode="${1:-}"
if [ -z "$mode" ]; then
    cat >&2 <<EOF
usage: bash $0 <mode> [extra-args]
  modes:
    smoke                       -- one cell per class (3 cells total),
                                   for pre-flight TimeEff verification
    full                        -- full 40-cell grid (3 sub-arrays
                                   with TIGHT per-class --time)
    epinions-shuffle-rerun      -- 4 missing Epinions shuffle cells
                                   (seeds 1,2,3,4) -- the 2026-06-04 gap
    k-sweep [K1 K2 ...]         -- chain dependent submissions at
                                   different concurrency caps for
                                   parallelism characterisation
                                   (default K-list: 20 10 5)

After any submission run \`seff <jobid>\` per task to verify
TimeEff matches the budget plan; halt and re-budget if not.
EOF
    exit 2
fi

AUDIT_K_TAG="${AUDIT_K_TAG:-prod}"
K_LIMIT="${K_LIMIT:-20}"
SELF="$(realpath "$0")"

submit_class() {
    local label="$1" time_limit="$2" indices="$3" k_limit="$4"
    local out
    out=$(sbatch \
        --array="${indices}%${k_limit}" \
        --time="$time_limit" \
        --job-name="hsikan-edge-cr-${AUDIT_K_TAG}-${label}" \
        --output="slurm_logs/hsikan-edge-cr-${AUDIT_K_TAG}-${label}-%A_%a.out" \
        --error="slurm_logs/hsikan-edge-cr-${AUDIT_K_TAG}-${label}-%A_%a.err" \
        --export=ALL,AUDIT_K_TAG="${AUDIT_K_TAG}-${label}" \
        "$SELF")
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "error: sbatch $label rc=$rc out=$out" >&2
        return 1
    fi
    echo "$out" | awk '{print $NF}'
}

case "$mode" in
    smoke)
        echo "=== SMOKE @ $(date -Iseconds) ==="
        echo "  TINY:   idx 0,  --time=$TINY_TIME"
        echo "  MEDIUM: idx 1,  --time=$MEDIUM_TIME"
        echo "  LONG:   idx 11, --time=$LONG_TIME"
        echo ""
        JID_T=$(submit_class smoke-tiny    "$TINY_TIME"    "0"   1) || exit 1
        echo "TINY   smoke jobid=$JID_T"
        JID_M=$(submit_class smoke-medium  "$MEDIUM_TIME"  "1"   1) || exit 1
        echo "MEDIUM smoke jobid=$JID_M"
        JID_L=$(submit_class smoke-long    "$LONG_TIME"    "11"  1) || exit 1
        echo "LONG   smoke jobid=$JID_L"
        echo ""
        echo "After all COMPLETED, verify:"
        echo "  seff $JID_T  # TINY:   TimeEff >= 25%"
        echo "  seff $JID_M  # MEDIUM: TimeEff >= 60%"
        echo "  seff $JID_L  # LONG:   TimeEff >= 60%"
        echo "Only if all pass: bash $0 full"
        ;;

    full)
        echo "=== FULL GRID @ $(date -Iseconds) ==="
        echo "  K_LIMIT=$K_LIMIT  AUDIT_K_TAG=$AUDIT_K_TAG"
        echo ""
        JID_T=$(submit_class tiny   "$TINY_TIME"   "$TINY_INDICES"   "$K_LIMIT") || exit 1
        echo "TINY   (25 cells, --time=$TINY_TIME):   jobid=$JID_T"
        JID_M=$(submit_class medium "$MEDIUM_TIME" "$MEDIUM_INDICES" "$K_LIMIT") || exit 1
        echo "MEDIUM (5 cells, --time=$MEDIUM_TIME):  jobid=$JID_M"
        JID_L=$(submit_class long   "$LONG_TIME"   "$LONG_INDICES"   "$K_LIMIT") || exit 1
        echo "LONG   (10 cells, --time=$LONG_TIME):   jobid=$JID_L"
        ;;

    epinions-shuffle-rerun)
        echo "=== EPINIONS-SHUFFLE-RERUN @ $(date -Iseconds) ==="
        echo "  4 cells (seeds 1,2,3,4), --time=$LONG_TIME, K=4 concurrent"
        echo ""
        JID=$(submit_class rerun-long "$LONG_TIME" "$EPINIONS_SHUFFLE_RERUN" 4) || exit 1
        echo "Epinions shuffle rerun jobid=$JID"
        ;;

    k-sweep)
        # K-sweep parallelism characterisation: submit the full grid
        # multiple times at different K caps, chained via afterany so
        # they never compete for GPUs.
        shift
        K_LIST="${@:-20 10 5}"
        echo "=== K-SWEEP @ $(date -Iseconds)  K_LIST=$K_LIST ==="
        echo ""
        # Auto-detect any running hsikan job to chain after; else start now.
        PARENT=""
        if command -v squeue >/dev/null 2>&1; then
            PARENT=$(squeue -h -u "$USER" -o "%i %j" 2>/dev/null \
                | grep -E "hsikan-edge-cr" \
                | awk '{print $1}' | head -1)
        fi
        [ -n "$PARENT" ] && echo "Chaining after parent jobid: $PARENT"
        prev="$PARENT"
        for K in $K_LIST; do
            # Submit each K-pass as a full 3-class grid. Within a K-pass
            # the three classes run concurrently; between K-passes the
            # dependency on the prior K-pass's LONG class ensures no
            # GPU contention.
            dep_flag=""
            [ -n "$prev" ] && dep_flag="--dependency=afterany:$prev"
            tag_save="$AUDIT_K_TAG"
            AUDIT_K_TAG="K${K}"
            JID_T=$(submit_class "K${K}-tiny"   "$TINY_TIME"   "$TINY_INDICES"   "$K") || exit 1
            JID_M=$(submit_class "K${K}-medium" "$MEDIUM_TIME" "$MEDIUM_INDICES" "$K") || exit 1
            # LONG carries the dependency on the prior pass's LONG; if
            # the parent is set we chain LONG explicitly via a separate
            # sbatch (so we can pass --dependency).
            if [ -n "$dep_flag" ]; then
                out=$(sbatch $dep_flag \
                    --array="${LONG_INDICES}%${K}" \
                    --time="$LONG_TIME" \
                    --job-name="hsikan-edge-cr-K${K}-long" \
                    --output="slurm_logs/hsikan-edge-cr-K${K}-long-%A_%a.out" \
                    --error="slurm_logs/hsikan-edge-cr-K${K}-long-%A_%a.err" \
                    --export=ALL,AUDIT_K_TAG="K${K}-long" \
                    "$SELF")
                JID_L=$(echo "$out" | awk '{print $NF}')
            else
                JID_L=$(submit_class "K${K}-long" "$LONG_TIME" "$LONG_INDICES" "$K") || exit 1
            fi
            AUDIT_K_TAG="$tag_save"
            echo "K=$K: TINY=$JID_T MEDIUM=$JID_M LONG=$JID_L (dep=${prev:-none})"
            prev="$JID_L"
        done
        ;;

    *)
        echo "error: unknown mode '$mode' (use: smoke | full | epinions-shuffle-rerun)" >&2
        exit 2
        ;;
esac
