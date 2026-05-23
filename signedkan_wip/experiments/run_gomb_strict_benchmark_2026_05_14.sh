#!/usr/bin/env bash
# Gömb strict-protocol overnight benchmark with PER-DATASET OPTUNA-TUNED CONFIGS.
#
# Background: post-ChatGPT-audit (2026-05-14), Gömb is the only one of our
# architectures that runs strict protocol by construction (cycle enumeration
# uses TRAIN edges only via run_gomb_smoke). HSiKAN's published numbers
# include transductive σ-leakage; Gömb's are the honest architectural baseline.
#
# Per-dataset configs come from PRIOR GÖMB OPTUNA RUNS (reused, not refit):
#
#   Bitcoin Alpha  ← gomb_tune_sota_chase_alpha_joint_2026_05_12.jsonl
#                    M_outer=8, d_outer=20, d_middle=24, d_core=48,
#                    n_tiers=4, topk=56, lr=0.005  (single-seed test 0.9081)
#
#   Bitcoin OTC    ← gomb_tune_joint_run.jsonl
#                    M_outer=12, d_outer=8, d_middle=16, d_core=32,
#                    n_tiers=2, topk=32, lr=0.005  (single-seed test 0.9238)
#
#   Slashdot       ← reports/2026-05-11-hymeko-gomb-slashdot-sota-attempt.md
#                    d_embed=16, M_outer=4, d_outer=4, d_middle=4,
#                    d_core=4, n_tiers=3, topk=32, lr=3e-3, n_epochs=60
#                    (5-seed 0.9031 ± 0.0008 — the published "slim" config)
#
#   Epinions       ← no prior Gömb-Optuna; reuse Slashdot slim config (same
#                    large-graph regime). Conservative starting point.
#
# Step 0 (label-shuffle confirmation) already completed in the previous run
# at gomb_strict_benchmark_20260514T005336Z/step0_shuffle_alpha_seed0.log
# (val=0.5692, test=0.5402 — Gömb is confirmed strict). Skipping here.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

# CUDA fragmentation control — Gömb's cycle pool allocates many small tensors.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/gomb_strict_benchmark_tuned_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"

echo "=== Gömb strict-protocol overnight benchmark (TUNED configs) ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

# Common runner — extra args at the end override defaults.
run_gomb() {
    local label="$1"; shift
    local dataset="$1"; shift
    local seed="$1"; shift
    local epochs="$1"; shift
    local extra_args="$@"
    local outf="${OUT_DIR}/${label}_seed${seed}.log"
    echo "[$(date -Is)] START $label seed=$seed dataset=$dataset epochs=$epochs extra='$extra_args'" \
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
# Step 1 — Bitcoin Alpha 5-seed (Optuna-tuned)
#   M_outer=8 d_outer=20 d_middle=24 d_core=48 n_tiers=4 topk=56 lr=0.005
# =================================================================
echo "## Step 1 — Bitcoin Alpha 5-seed strict (tuned: M_outer=8 d_outer=20 d_middle=24 d_core=48 n_tiers=4 topk=56 lr=5e-3)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step1_alpha" "bitcoin_alpha" $seed 80 \
        --M-outer 8 --d-outer 20 --d-middle 24 --d-core 48 \
        --n-tiers 4 --topk 56 --lr 0.005
done

# =================================================================
# Step 2 — Bitcoin OTC 5-seed (Optuna-tuned)
#   M_outer=12 d_outer=8 d_middle=16 d_core=32 n_tiers=2 topk=32 lr=0.005
# =================================================================
echo "## Step 2 — Bitcoin OTC 5-seed strict (tuned: M_outer=12 d_outer=8 d_middle=16 d_core=32 n_tiers=2 topk=32 lr=5e-3)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step2_otc" "bitcoin_otc" $seed 80 \
        --M-outer 12 --d-outer 8 --d-middle 16 --d-core 32 \
        --n-tiers 2 --topk 32 --lr 0.005
done

# =================================================================
# Step 3 — Slashdot 5-seed (published SOTA slim config)
#   d_embed=16 M_outer=4 d_outer=4 d_middle=4 d_core=4 n_tiers=3 topk=32 60 epochs
# =================================================================
echo "## Step 3 — Slashdot 5-seed strict (slim SOTA: d_embed=16 M_outer=4 d_outer=4 d_middle=4 d_core=4 n_tiers=3 topk=32 lr=3e-3)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step3_slashdot" "slashdot" $seed 60 \
        --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
        --n-tiers 3 --topk 32 --lr 0.003
done

# =================================================================
# Step 4 — Epinions 5-seed (slim config, same regime as Slashdot)
#   d_embed=16 M_outer=4 d_outer=4 d_middle=4 d_core=4 n_tiers=3 topk=32 60 epochs
# =================================================================
echo "## Step 4 — Epinions 5-seed strict (slim, same as Slashdot)" | tee -a "$LOG"
for seed in 0 1 2 3 4; do
    run_gomb "step4_epinions" "epinions" $seed 60 \
        --d-embed 16 --M-outer 4 --d-outer 4 --d-middle 4 --d-core 4 \
        --n-tiers 3 --topk 32 --lr 0.003
done

# =================================================================
# Final summary
# =================================================================
echo "" | tee -a "$LOG"
echo "=== Benchmark complete: $(date -Is) ===" | tee -a "$LOG"
echo "Results in: $OUT_DIR" | tee -a "$LOG"
