#!/bin/bash
# Master orchestration for the Epinions A+B+C+D lift sweep.
#
# Phases (each gated on prior best AUC):
#   A. Sub-unit c    : c ∈ {0.25, 0.5, 0.75}        (3 smokes)
#   B. Sparse attn   : K_attn ∈ {4, 8, 16}          (3 smokes at best A)
#   C. Davis pruner  : pruner=davis                  (1 smoke at best A+B)
#   D. m_max sweep   : m_max ∈ {32, 64}              (2 smokes at best A+B+C)
#   E. 5-seed paired: best-stack vs adaptive c=1     (10 runs)
#
# Plan:   docs/plans/2026-05-10-epinions-lift-studies/plan.tex
# Probe predicted: c=0.25 best (mean cycle score 0.738)
#
# Total wall budget:
#   Wait for in-flight 5-seed: ~3 h
#   Smokes A+B+C+D:            ~30-50 min
#   Final 5-seed:              ~3.8 h
#   Total:                     ~7-8 h
#
# Usage: bash signedkan_wip/experiments/run_lift_stack_2026_05_10.sh

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="/tmp/lift_stack_2026_05_10"
mkdir -p "$LOG_DIR"

# Decision log (markdown table; appended through the run).
DECISIONS="$LOG_DIR/decisions.md"
SMOKES="$LOG_DIR/smokes.tsv"
echo -e "phase\tparams\twall_s\tauc\tf1m" > "$SMOKES"
{
    echo "# A+B+C+D lift-stack decision log"
    echo
    echo "Started: $(date)"
    echo
} > "$DECISIONS"

# ─── Phase 0: wait for any in-flight 5-seed to clear the GPU ───────
echo "[lift-stack] $(date +%H:%M:%S) Phase 0: waiting for any in-flight 5-seed"
while pgrep -af "run_adaptive_mv_5seed_2026_05_10" \
        | grep -v "$$" | grep -v "lift_stack" | grep -q .; do
    sleep 60
done
echo "[lift-stack] $(date +%H:%M:%S) GPU is free, proceeding"

# Helper: run one smoke, append to SMOKES, echo result.
# args: phase params extra-env-pairs...
run_smoke() {
    local phase="$1" params="$2"
    shift 2
    local logf="$LOG_DIR/smoke_${phase}_${params// /_}.log"
    local t0=$(date +%s)
    echo "[lift-stack] $(date +%H:%M:%S) START phase=$phase $params"
    env "$@" \
        HSIKAN_MIXED_TUPLES=c3,c4 \
        python -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset epinions --hidden 4 --n-epochs 20 --seed 0 \
            > "$logf" 2>&1
    local elapsed=$(( $(date +%s) - t0 ))
    local r
    r=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$r" ]; then
        local auc f1
        auc=$(echo "$r" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        f1=$(echo "$r" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["f1m"], 4))')
        echo -e "${phase}\t${params}\t${elapsed}\t${auc}\t${f1}" >> "$SMOKES"
        echo "[lift-stack] $(date +%H:%M:%S) OK phase=$phase $params AUC=$auc wall=${elapsed}s"
    else
        echo "[lift-stack] $(date +%H:%M:%S) FAIL phase=$phase $params (see $logf)"
    fi
}

# Helper: pick the params with the highest AUC in a given phase.
best_in_phase() {
    local phase="$1"
    awk -F'\t' -v p="$phase" 'NR>1 && $1==p { if ($4 > best || NR_p++==0) { best=$4; bp=$2 } } END { print bp }' "$SMOKES"
}

# ─── Phase A: sub-unit c sweep ──────────────────────────────────────
echo "[lift-stack] === Phase A: sub-unit c ==="
for c in 0.25 0.5 0.75; do
    run_smoke A "c=$c" \
        HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=128 \
        HSIKAN_TOPK_M_V_MIN=1 \
        HSIKAN_TOPK_M_V_MAX=128 \
        HSIKAN_TOPK_M_V_C="$c" \
        HSIKAN_TOPK_PRUNER=balance \
        HSIKAN_TOPK_SCORER=fraction_negative
done
A_BEST=$(best_in_phase A)
A_C=$(echo "$A_BEST" | sed 's/c=//')
{ echo "## Phase A best: $A_BEST"; echo; } >> "$DECISIONS"
echo "[lift-stack] Phase A best: $A_BEST"

# ─── Phase B: sparse attention at best A ────────────────────────────
echo "[lift-stack] === Phase B: sparse attention at $A_BEST ==="
for k_attn in 4 8 16; do
    run_smoke B "c=$A_C K_attn=$k_attn" \
        HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=128 \
        HSIKAN_TOPK_M_V_MIN=1 \
        HSIKAN_TOPK_M_V_MAX=128 \
        HSIKAN_TOPK_M_V_C="$A_C" \
        HSIKAN_TOPK_PRUNER=balance \
        HSIKAN_TOPK_SCORER=fraction_negative \
        HSIKAN_ATTENTION_M_E=quaternion \
        HSIKAN_SPARSE_ATTN_K="$k_attn"
done
B_BEST=$(best_in_phase B)
B_K=$(echo "$B_BEST" | sed 's/.*K_attn=//')
{ echo "## Phase B best: $B_BEST"; echo; } >> "$DECISIONS"
echo "[lift-stack] Phase B best: $B_BEST"

# Decide whether B helps (compare best B AUC to best A AUC)
A_AUC=$(awk -F'\t' -v p="$A_BEST" 'NR>1 && $1=="A" && $2==p {print $4}' "$SMOKES")
B_AUC=$(awk -F'\t' -v p="$B_BEST" 'NR>1 && $1=="B" && $2==p {print $4}' "$SMOKES")
if python -c "import sys; sys.exit(0 if float('$B_AUC') > float('$A_AUC') else 1)"; then
    USE_SPARSE_ATTN="HSIKAN_SPARSE_ATTN_K=$B_K"
    USE_QUAT="HSIKAN_ATTENTION_M_E=quaternion"
    { echo "Sparse attn HELPS (+$(python -c "print(round(float('$B_AUC') - float('$A_AUC'), 4))")); using K_attn=$B_K"; echo; } >> "$DECISIONS"
else
    USE_SPARSE_ATTN=""
    USE_QUAT=""
    { echo "Sparse attn does NOT help; dropping it"; echo; } >> "$DECISIONS"
fi

# ─── Phase C: Davis pruner ──────────────────────────────────────────
echo "[lift-stack] === Phase C: Davis pruner ==="
run_smoke C "c=$A_C pruner=davis" \
    HSIKAN_TOPK_MODE=per_vertex_adaptive \
    HSIKAN_TOPK_K=128 \
    HSIKAN_TOPK_M_V_MIN=1 \
    HSIKAN_TOPK_M_V_MAX=128 \
    HSIKAN_TOPK_M_V_C="$A_C" \
    HSIKAN_TOPK_PRUNER=davis \
    HSIKAN_TOPK_SCORER=fraction_negative \
    ${USE_QUAT:+$USE_QUAT} \
    ${USE_SPARSE_ATTN:+$USE_SPARSE_ATTN}

C_AUC=$(awk -F'\t' 'NR>1 && $1=="C" {print $4}' "$SMOKES")
PHASE_C_BEST_AUC=$(python -c "print(max(float('$A_AUC'), float('$B_AUC' or '0')))")
if python -c "import sys; sys.exit(0 if float('$C_AUC') > float('$PHASE_C_BEST_AUC') else 1)"; then
    USE_PRUNER="davis"
    { echo "Davis HELPS; using pruner=davis"; echo; } >> "$DECISIONS"
else
    USE_PRUNER="balance"
    { echo "Davis does NOT help; sticking with balance"; echo; } >> "$DECISIONS"
fi

# ─── Phase D: m_max sweep ───────────────────────────────────────────
echo "[lift-stack] === Phase D: m_max sweep ==="
for m_max in 32 64; do
    run_smoke D "c=$A_C m_max=$m_max pruner=$USE_PRUNER" \
        HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=128 \
        HSIKAN_TOPK_M_V_MIN=1 \
        HSIKAN_TOPK_M_V_MAX="$m_max" \
        HSIKAN_TOPK_M_V_C="$A_C" \
        HSIKAN_TOPK_PRUNER="$USE_PRUNER" \
        HSIKAN_TOPK_SCORER=fraction_negative \
        ${USE_QUAT:+$USE_QUAT} \
        ${USE_SPARSE_ATTN:+$USE_SPARSE_ATTN}
done
D_BEST=$(best_in_phase D)
D_M_MAX=$(echo "$D_BEST" | sed 's/.*m_max=\([0-9]*\).*/\1/')
{ echo "## Phase D best: $D_BEST"; echo; } >> "$DECISIONS"
D_AUC=$(awk -F'\t' -v p="$D_BEST" 'NR>1 && $1=="D" && $2==p {print $4}' "$SMOKES")
if python -c "import sys; sys.exit(0 if float('$D_AUC') > float('$PHASE_C_BEST_AUC') and float('$D_AUC') > float('$C_AUC') else 1)"; then
    FINAL_M_MAX="$D_M_MAX"
    { echo "m_max=$D_M_MAX HELPS; using it"; echo; } >> "$DECISIONS"
else
    FINAL_M_MAX=128
    { echo "m_max sweep does NOT help; sticking with 128"; echo; } >> "$DECISIONS"
fi

# ─── Phase E: final 5-seed paired ───────────────────────────────────
echo "[lift-stack] === Phase E: 5-seed paired (best stack vs adaptive c=1) ==="
RESULTS_5SEED="$LOG_DIR/results_5seed.tsv"
echo -e "variant\tseed\twall_s\tauc\tf1m" > "$RESULTS_5SEED"

run_5seed() {
    local variant="$1" seed="$2"
    shift 2
    local logf="$LOG_DIR/5seed_${variant}_seed${seed}.log"
    local t0=$(date +%s)
    echo "[lift-stack] $(date +%H:%M:%S) START 5seed variant=$variant seed=$seed"
    env "$@" \
        HSIKAN_MIXED_TUPLES=c2,c3,c4,c5,w2,w3 \
        HSIKAN_ATTENTION_HIGHWAY=1 \
        HSIKAN_ATTENTION_HIGHWAY_KIND=edge_cr \
        HSIKAN_CYCLE_BATCH=2000 \
        HSIKAN_TRITON_KERNEL=1 \
        HSIKAN_TRITON_BACKWARD=1 \
        HYMEKO_CYCLE_CACHE=1 \
        python -m signedkan_wip.experiments.runs.run_final_cell \
            --dataset epinions --hidden 4 --n-epochs 80 \
            --max-k4 100000 --seed "$seed" \
            > "$logf" 2>&1
    local elapsed=$(( $(date +%s) - t0 ))
    local r
    r=$(grep -E '^\{"dataset"' "$logf" | tail -1)
    if [ -n "$r" ]; then
        local auc f1
        auc=$(echo "$r" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["auc"], 4))')
        f1=$(echo "$r" | python -c 'import sys,json;print(round(json.loads(sys.stdin.read())["f1m"], 4))')
        echo -e "${variant}\t${seed}\t${elapsed}\t${auc}\t${f1}" >> "$RESULTS_5SEED"
        echo "[lift-stack] $(date +%H:%M:%S) OK 5seed variant=$variant seed=$seed AUC=$auc"
    else
        echo "[lift-stack] $(date +%H:%M:%S) FAIL 5seed variant=$variant seed=$seed"
    fi
}

for seed in 0 1 2 3 4; do
    # Adaptive c=1 reference (matches the in-flight baseline if it
    # already ran, otherwise we get a fresh paired anchor)
    run_5seed "adaptive_c1" "$seed" \
        HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=128 \
        HSIKAN_TOPK_M_V_MIN=1 \
        HSIKAN_TOPK_M_V_MAX=128 \
        HSIKAN_TOPK_M_V_C=1.0 \
        HSIKAN_TOPK_PRUNER=balance \
        HSIKAN_TOPK_SCORER=fraction_negative \
        HSIKAN_ATTENTION_M_E=quaternion

    # Best stack
    run_5seed "best_stack" "$seed" \
        HSIKAN_TOPK_MODE=per_vertex_adaptive \
        HSIKAN_TOPK_K=128 \
        HSIKAN_TOPK_M_V_MIN=1 \
        HSIKAN_TOPK_M_V_MAX="$FINAL_M_MAX" \
        HSIKAN_TOPK_M_V_C="$A_C" \
        HSIKAN_TOPK_PRUNER="$USE_PRUNER" \
        HSIKAN_TOPK_SCORER=fraction_negative \
        HSIKAN_ATTENTION_M_E=quaternion \
        ${USE_SPARSE_ATTN:+$USE_SPARSE_ATTN}
done

echo
echo "=== Smoke phase summary ==="
column -t -s $'\t' "$SMOKES"
echo
echo "=== 5-seed summary ==="
column -t -s $'\t' "$RESULTS_5SEED"

# Paired stats
python - <<PY
import csv, statistics
from collections import defaultdict
with open("$RESULTS_5SEED") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
by = defaultdict(list)
for r in rows:
    by[r["variant"]].append(r)
print()
print("─── Final 5-seed paired stats ───")
for variant in ("adaptive_c1", "best_stack"):
    rs = sorted(by.get(variant, []), key=lambda r: int(r["seed"]))
    if len(rs) != 5:
        print(f"  {variant}: incomplete ({len(rs)}/5)")
        continue
    aucs = [float(r["auc"]) for r in rs]
    am, asd = statistics.mean(aucs), statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    walls = [float(r["wall_s"]) for r in rs]
    wm = statistics.mean(walls)
    per = ", ".join(r["auc"] for r in rs)
    print(f"  {variant:<14}  AUC={am:.4f} ± {asd:.4f}  wall={wm:.0f}s  per-seed: {per}")

a = sorted(by.get("adaptive_c1", []), key=lambda r: int(r["seed"]))
b = sorted(by.get("best_stack",   []), key=lambda r: int(r["seed"]))
if len(a) == 5 and len(b) == 5:
    deltas = [float(bb["auc"]) - float(aa["auc"]) for aa, bb in zip(a, b)]
    dm = statistics.mean(deltas)
    ds = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    sigma = (dm * (5 ** 0.5)) / max(ds, 1e-9)
    print()
    print(f"  Paired (best_stack - adaptive_c1):  Δ={dm:+.4f} ± {ds:.4f}  σ={sigma:+.2f}")
    per = ", ".join(f"{d:+.4f}" for d in deltas)
    print(f"  Per-seed Δ: {per}")
PY

echo
echo "[lift-stack] $(date +%H:%M:%S) DONE"
