#!/usr/bin/env bash
# Ricci-Stim Cluttered MNIST overnight ablation — 5 configs × 1 seed.
#
# Each config trains RicciStimDetector for 20 epochs on 5000 train +
# 1000 eval Cluttered MNIST images. Generates the full ablation table:
#   A: bare backbone (α=0, β=0, no SDRF) — Phase 7 baseline
#   B: Hodge smoothing alone (α=0.1, β=0)
#   C: Ricci correction alone (α=0, β=0.1)
#   D: Bochner full (α=0.1, β=0.1)
#   E: headline — Bochner + SDRF rewiring
#
# Target: Config E ≥ 0.5 mAP50 = stimulus-driven hypothesis confirmed.
# Reference: HyMeYOLO +ricci-mod = 0.723 mAP50 on the same task.
# Estimated wall: ~5 hours.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/ricci_stim_ablation_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
JSONL="${OUT_DIR}/ablation.jsonl"

echo "=== Ricci-Stim Cluttered MNIST ablation (5 configs × 1 seed) ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"
echo | tee -a "$LOG"

for CFG in A B C D E; do
    CFG_LOG="${OUT_DIR}/config_${CFG}.log"
    echo "[$(date -Is)] START config=$CFG" | tee -a "$LOG"
    t0=$(date +%s)
    python -m signedkan_wip.experiments.run_ricci_stim_cluttered_mnist \
        --config "$CFG" --n-train 5000 --n-eval 1000 --n-epochs 20 \
        --seed 0 --device cuda --batch-size 8 \
        --out-jsonl "$JSONL" \
        > "$CFG_LOG" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    final_map=$(grep -oE "final_mAP50_proxy=[0-9.]+" "$CFG_LOG" | tail -1 | cut -d= -f2)
    echo "[$(date -Is)] DONE  config=$CFG rc=$rc elapsed=${elapsed}s mAP50_proxy=${final_map}" | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "=== Ablation summary ===" | tee -a "$LOG"
JSONL="$JSONL" python3 <<'EOF' | tee -a "$LOG"
import json, os
rows = []
with open(os.environ["JSONL"]) as f:
    for line in f:
        rows.append(json.loads(line))
print(f"{'Config':<8} {'α':>4} {'β':>4} {'SDRF':>5} {'wall(s)':>8} {'mAP50_proxy':>12}")
for r in rows:
    cfg = r["config"]
    a = r["ablation_settings"]["bochner_alpha"]
    b = r["ablation_settings"]["bochner_beta"]
    s = r["ablation_settings"]["use_sdrf"]
    print(f"{cfg:<8} {a:>4} {b:>4} {str(s):>5} {r['wall_s']:>8.1f} {r['final_mAP50_proxy']:>12.4f}")
EOF

echo "end: $(date -Is)" | tee -a "$LOG"
echo "results: $OUT_DIR" | tee -a "$LOG"
