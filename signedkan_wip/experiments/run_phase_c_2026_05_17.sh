#!/usr/bin/env bash
# Phase C measurement — Nature Comm submission track (2026-05-17).
#
# Two purposes:
#   1. Run Gömb-strict 5-seed on the NEW Reddit Hyperlinks dataset
#      (NOT in SE-SGformer or DADSGNN coverage — the "they didn't
#      optimise against this" test the Nature Comm plan §C calls for).
#   2. Run Gömb-strict on 5 synthetic SBM/hier configs (balance sweep),
#      with the label-shuffle audit applied to each config as a positive
#      control (chance-AUC expected under the strict protocol; any
#      method that holds above chance has a leakage bug).
#
# Phase C plan: docs/plans/2026-05-17-nature-comm-leakage-audit/plan.tex §C.
#
# Total expected wall: ~4-5 hours on sole-GPU.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/phase_c_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Phase C: Reddit + synthetic SBM/hier balance sweep ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Shared runner helper.
run_gomb() {
    local label="$1"; shift
    local dataset="$1"; shift
    local seed="$1"; shift
    local epochs="$1"; shift
    local extra_args="$@"
    local outf="${OUT_DIR}/${label}_seed${seed}.log"
    echo "[$(date -Is)] START $label seed=$seed dataset=$dataset epochs=$epochs" \
        | tee -a "$LOG"
    local t0=$(date +%s)
    python -m signedkan_wip.experiments.runs.run_gomb_smoke \
        --dataset "$dataset" --seed "$seed" \
        --n-epochs "$epochs" \
        --edge-split 80_10_10 --joint-mix \
        --device cuda \
        $extra_args \
        > "$outf" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    local result
    result=$(grep -E '^\{"dataset"' "$outf" | tail -1)
    if [ -z "$result" ]; then
        result=$(grep -E "val_auroc|test_auroc|OutOfMemoryError|Error" "$outf" | tail -1)
    fi
    echo "[$(date -Is)] DONE  $label seed=$seed rc=$rc elapsed=${elapsed}s" \
        | tee -a "$LOG"
    echo "  result: $result" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

# =================================================================
# Step 1 — Reddit Hyperlink Network (the "they didn't optimise against this" dataset)
# Use Slashdot slim Optuna-best config (closest large-graph regime).
# =================================================================
echo "## Step 1 — Reddit Hyperlinks (title) 5-seed strict (slim Optuna config)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step1_reddit_title" "reddit_title" $seed 60 \
        --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
        --n-tiers 3 --topk 32 --lr 0.003
done

# Label-shuffle audit on Reddit (one seed for positive-control check)
echo "## Step 1b — Reddit shuffle audit (seed=0)" | tee -a "$LOG"
run_gomb "step1b_reddit_shuffle" "reddit_title" 0 60 \
    --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
    --n-tiers 3 --topk 32 --lr 0.003 --shuffle-train-signs

# =================================================================
# Step 2-6 — Synthetic SBM/hier balance sweep
# Small graphs; use moderate config.
# =================================================================
SYNTH_CONFIG=(--M-outer 4 --d-outer 8 --d-middle 8 --d-core 8 \
              --n-tiers 2 --topk 16 --lr 0.005)

echo "## Step 2 — Synthetic SBM k=4 pos_in=0.85 (balanced baseline)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step2_sbm_balanced" "sbm_n200_k4_s${seed}" $seed 60 ${SYNTH_CONFIG[@]}
done

echo "## Step 3 — Synthetic SBM pos_in=0.60 (unbalanced)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step3_sbmsweep_pos60" "sbmsweep_pos60_s${seed}" $seed 60 ${SYNTH_CONFIG[@]}
done

echo "## Step 4 — Synthetic SBM pos_in=0.75 (mildly unbalanced)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step4_sbmsweep_pos75" "sbmsweep_pos75_s${seed}" $seed 60 ${SYNTH_CONFIG[@]}
done

echo "## Step 5 — Synthetic SBM pos_in=0.95 (strongly balanced)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step5_sbmsweep_pos95" "sbmsweep_pos95_s${seed}" $seed 60 ${SYNTH_CONFIG[@]}
done

echo "## Step 6 — Synthetic hierarchical k=4" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step6_hier" "hier_n240_s${seed}" $seed 60 ${SYNTH_CONFIG[@]}
done

# Shuffle audit on synthetic SBM baseline (positive control)
echo "## Step 7 — Shuffle audit on SBM balanced (seed=0)" | tee -a "$LOG"
run_gomb "step7_sbm_balanced_shuffle" "sbm_n200_k4_s0" 0 60 \
    ${SYNTH_CONFIG[@]} --shuffle-train-signs

echo "[$(date -Is)] Phase C complete." | tee -a "$LOG"
echo "Results dir: $OUT_DIR"
