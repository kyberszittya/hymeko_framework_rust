#!/usr/bin/env bash
# ============================================================================
# 2026-05-11 evening overnight
#
#   Stage 0  wait for current synthetic HymeYOLO to clear
#   Stage 1  HymeYOLO PASCAL VOC 5-seed
#   Stage 2  Gömb multi-dataset matrix
#              4 tiers × {full Gömb[k=3], MixedArity[3,4]} × 5 seeds
#   Stage 3  Slashdot-only: MixedArity[4,5] 5-seed (SOTA-break attempt)
#
# Memory caps follow the 2026-05-11 CLAUDE.md §4 amendment: NO `ulimit -v`
# (breaks PyTorch+CUDA at first .to(cuda)). Rely on host RSS headroom +
# the 8 GB GPU's own physical cap. systemd-run cgroups would tighten if
# needed.
#
# Artifacts:
#   $LOGDIR/MASTER.log      timestamped human-readable progress
#   $LOGDIR/voc_s${s}.{log,jsonl,err}
#   $LOGDIR/{gomb_k3,mixed34,mixed45}_${ds}_s${s}.log
#   $LOGDIR/results.jsonl  one JSON line per gomb run, appended
# ============================================================================
set -uo pipefail

# Move to repo root regardless of where the user fired this from.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOGDIR=reports/overnight_2026_05_11_voc_gomb_matrix
MASTER="$LOGDIR/MASTER.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER"; }

log "=== VOC + Gömb matrix overnight START (PID $$) ==="
log "git SHA $(git rev-parse HEAD)"

# ----------------------------------------------------------------------------
# Stage 0: wait for prior HymeYOLO synthetic to clear.
# ----------------------------------------------------------------------------
if pgrep -f train_circles_ricci > /dev/null 2>&1; then
    log "Stage 0: waiting for in-flight synthetic HymeYOLO to clear"
    while pgrep -f train_circles_ricci > /dev/null 2>&1; do sleep 30; done
fi
log "Stage 0: GPU free"

# ----------------------------------------------------------------------------
# Stage 1: HymeYOLO PASCAL VOC 5-seed
# ----------------------------------------------------------------------------
log "=== Stage 1: HymeYOLO PASCAL VOC 5-seed ==="
for s in 0 1 2 3 4; do
    log "Stage 1: voc_s${s} START"
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -u -m signedkan_wip.src.vision.train_voc \
          --year 2007 --image-set trainval \
          --input-size 128 --epochs 30 --batch-size 16 --seed "$s" \
          --jsonl-out "$LOGDIR/voc_s${s}.jsonl" \
          > "$LOGDIR/voc_s${s}.log" \
          2> "$LOGDIR/voc_s${s}.err"
    log "Stage 1: voc_s${s} DONE rc=$?"
done

# ----------------------------------------------------------------------------
# Stage 2: Gömb / MixedArity multi-dataset paired matrix
# ----------------------------------------------------------------------------
# Shared slim config (Slashdot 5-seed winner: 0.9031 ± 0.0008). Apply
# uniformly so cross-dataset numbers are directly comparable.
SLIM=(
    --d-embed 16 --d-outer 4 --M-outer 4
    --d-middle 4 --d-core 4 --n-tiers 3
    --n-epochs 50
)

# Args: dataset device topk -> runs gomb[k=3], mixed[3,4] paired 5-seed
run_pair() {
    local ds=$1 dev=$2 topk=$3
    log "Stage 2: dataset=$ds device=$dev topk=$topk START"
    for s in 0 1 2 3 4; do
        python -u -m signedkan_wip.src.run_gomb_smoke \
            --dataset "$ds" --seed "$s" --device "$dev" \
            "${SLIM[@]}" --topk "$topk" --k 3 \
            > "$LOGDIR/gomb_k3_${ds}_s${s}.log" 2>&1
        tail -1 "$LOGDIR/gomb_k3_${ds}_s${s}.log" >> "$LOGDIR/results.jsonl"

        python -u -m signedkan_wip.src.run_gomb_smoke \
            --dataset "$ds" --seed "$s" --device "$dev" \
            "${SLIM[@]}" --topk "$topk" --cycle-ks 3,4 \
            > "$LOGDIR/mixed34_${ds}_s${s}.log" 2>&1
        tail -1 "$LOGDIR/mixed34_${ds}_s${s}.log" >> "$LOGDIR/results.jsonl"
    done
    log "Stage 2: $ds 5-seed pair DONE"
}

log "=== Stage 2: Gömb 4-tier matrix ==="
# Tier A — synthetic control (architectural lever validation, fast on CPU)
run_pair sbm_n200_k4_s0  cpu  32
# Tier B — small real (iso-protocol with literature baselines; CPU-friendly)
run_pair bitcoin_alpha   cpu  32
run_pair bitcoin_otc     cpu  32
run_pair wiki_elec       cpu  32
# Tier C — medium real (the SOTA-break target on slashdot)
run_pair wikisigned      cuda 32
run_pair slashdot        cuda 32
# Tier D — heavy real (Epinions-scale; topk halved for memory safety)
run_pair epinions        cuda 16
run_pair wiki_conflict   cuda 16

# ----------------------------------------------------------------------------
# Stage 3: Slashdot-specific SOTA-break #2 attempt — mixed[4,5]
# Memory project_phase9_k45_sweet_spot: k=4+k=5 beats k=3+k=4 on every
# signed dataset. Single Slashdot 5-seed extension.
# ----------------------------------------------------------------------------
log "=== Stage 3: Slashdot mixed[4,5] 5-seed (SOTA-break attempt) ==="
for s in 0 1 2 3 4; do
    python -u -m signedkan_wip.src.run_gomb_smoke \
        --dataset slashdot --seed "$s" --device cuda \
        "${SLIM[@]}" --topk 16 --cycle-ks 4,5 \
        > "$LOGDIR/mixed45_slashdot_s${s}.log" 2>&1
    tail -1 "$LOGDIR/mixed45_slashdot_s${s}.log" >> "$LOGDIR/results.jsonl"
done
log "Stage 3: Slashdot mixed[4,5] DONE"

# ----------------------------------------------------------------------------
# Final: aggregate
# ----------------------------------------------------------------------------
python - <<'PY' "$LOGDIR/results.jsonl" "$LOGDIR/SUMMARY.txt" "$MASTER"
import json, statistics, sys, math
results_path, summary_path, master_path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for ln in open(results_path):
    ln = ln.strip()
    if not ln or not ln.startswith("{"): continue
    try:
        rows.append(json.loads(ln))
    except json.JSONDecodeError: pass

# Group by (dataset, model_label)
buckets = {}
for r in rows:
    key = (r["dataset"], r["model"])
    buckets.setdefault(key, []).append(r["val_auc_best"])

lines = ["dataset                  model                     n   mean    std     per-seed"]
lines.append("-" * 90)
for (ds, model), aucs in sorted(buckets.items()):
    if len(aucs) < 2:
        lines.append(f"{ds:24s} {model:24s}  {len(aucs):>2d}   {aucs[0]:.4f}   —       {aucs}")
        continue
    m = statistics.mean(aucs); sd = statistics.stdev(aucs)
    lines.append(f"{ds:24s} {model:24s}  {len(aucs):>2d}   {m:.4f}  {sd:.4f}   {[round(a,4) for a in aucs]}")

# Per-dataset paired Δ (mixed_arity vs gomb[k=3])
lines.append("")
lines.append("Paired Δ (per dataset, matched by seed): mixed_arity[3,4] − gomb[k=3]")
lines.append("-" * 90)
by_ds = {}
for r in rows:
    by_ds.setdefault(r["dataset"], {}).setdefault(r["model"], {})[r["seed"]] = r["val_auc_best"]
for ds, by_model in sorted(by_ds.items()):
    gomb = by_model.get("gomb", {})
    mixed = next((by_model[m] for m in by_model if m.startswith("mixed_arity_gomb[3,4")), {})
    common = sorted(set(gomb) & set(mixed))
    if len(common) < 2: continue
    deltas = [mixed[s] - gomb[s] for s in common]
    md = statistics.mean(deltas); sdv = statistics.stdev(deltas)
    se = sdv / math.sqrt(len(deltas)) if sdv > 0 else float("nan")
    sigma = md / se if se and se == se else float("nan")
    lines.append(f"  {ds:22s} Δ={md:+.4f}  σ={sigma:+.2f}  n={len(deltas)}  per_seed={[round(d,4) for d in deltas]}")

with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
with open(master_path, "a") as f:
    f.write("\n=== SUMMARY ===\n" + "\n".join(lines) + "\n")
PY

log "=== overnight done ==="
