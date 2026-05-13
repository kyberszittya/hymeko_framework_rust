#!/usr/bin/env bash
# Backfill the missing seeds 1, 2, 3 of the 2026-05-11 stage7 HyMeYOLO
# phase-1 5-seed sweep on Cluttered MNIST.
#
# Seeds 0 and 4 completed (after the ulimit -v fix at ~18:00 on 2026-05-11):
#   reports/overnight_2026_05_11_stage7/hymeyolo_ricci_n5k_e50_s{0,4}.jsonl
# Seeds 1, 2, 3 were silently killed earlier in that overnight run by
# `ulimit -v 16G` on CUDA workloads (29 GB VAS, healthy 1.77 GB RSS).
# See memory `feedback_ulimit_vs_cuda`.
#
# This script:
#   - queues behind any in-flight signedkan_wip.src.run_optuna_search or
#     run_final_cell process (Bitcoin 10-seed, future Slashdot Optuna);
#   - runs train_circles_ricci for seeds 1, 2, 3 at n=5000 / epochs=50;
#   - uses systemd-run --user -p MemoryMax=16G (cgroup v2 RSS cap), NOT
#     ulimit -v (CLAUDE.md §4, anti-pattern #10);
#   - writes jsonl to reports/overnight_2026_05_11_stage7/ so the redone
#     seeds sit alongside the existing 0/4 for aggregate analysis.
#
# Wall budget (estimated from seed 0/4 walls): ~30 min/seed × 3 = ~90 min.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# Use the env where torch lives; signedkan_wip imports work from either.
# The .venv has the CORE-pinned torch 2.4.1; miniconda3 has 2.11.
# Stage-7 ran under miniconda3 — match that for protocol parity.
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

OUT_DIR="reports/overnight_2026_05_11_stage7"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER="$OUT_DIR/REDO_seeds123_${STAMP}.master.log"
echo "[$(date -Is)] === stage7 seed 1/2/3 redo START stamp=$STAMP ===" > "$MASTER"
echo "git SHA: $(git rev-parse HEAD)" >> "$MASTER"

# Wait for the GPU to be free of signedkan_wip processes.
echo "[$(date -Is)] waiting for run_optuna_search / run_final_cell to clear..." | tee -a "$MASTER"
SELF_PID="$$"
while pgrep -af 'signedkan_wip\.src\.run_optuna_search|signedkan_wip\.src\.run_final_cell' \
      | grep -vE "^${SELF_PID} |^$ " | grep -vF "$0" | grep -q .; do
  sleep 60
done
echo "[$(date -Is)] GPU free; starting HyMeYOLO redo" | tee -a "$MASTER"

run_seed() {
  local seed="$1"
  local jsonl_out="$OUT_DIR/hymeyolo_ricci_n5k_e50_s${seed}.redo_${STAMP}.jsonl"
  local stdout_log="$OUT_DIR/hymeyolo_ricci_n5k_e50_s${seed}.redo_${STAMP}.json"
  local stderr_log="$OUT_DIR/hymeyolo_ricci_n5k_e50_s${seed}.redo_${STAMP}.err"
  local t0
  t0=$(date +%s)
  echo "[$(date -Is)] START seed=$seed jsonl=$jsonl_out" | tee -a "$MASTER"

  local cmd=(python -m signedkan_wip.src.vision.train_circles_ricci
             --n-images 5000 --epochs 50 --lr 0.003
             --seed "$seed" --jsonl-out "$jsonl_out")

  if command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope --quiet \
      -p MemoryMax=16G -p MemorySwapMax=0 \
      timeout 5400 "${cmd[@]}" \
      > "$stdout_log" 2> "$stderr_log"
  else
    timeout 5400 "${cmd[@]}" \
      > "$stdout_log" 2> "$stderr_log"
  fi
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ -s "$jsonl_out" ]; then
    local nrows
    nrows=$(wc -l < "$jsonl_out")
    echo "[$(date -Is)] OK    seed=$seed rc=$rc rows=$nrows elapsed=${elapsed}s" \
      | tee -a "$MASTER"
  else
    echo "[$(date -Is)] FAIL  seed=$seed rc=$rc elapsed=${elapsed}s (no jsonl rows)" \
      | tee -a "$MASTER"
    tail -5 "$stderr_log" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER"
  fi
}

for SEED in 1 2 3; do
  run_seed "$SEED"
done

# Aggregate across all 5 seeds (existing 0/4 + redone 1/2/3) per variant.
python - <<'PY'
import json, pathlib, statistics
out_dir = pathlib.Path("reports/overnight_2026_05_11_stage7")
rows = []
for f in sorted(out_dir.glob("hymeyolo_ricci_n5k_e50_s*.jsonl")) + \
         sorted(out_dir.glob("hymeyolo_ricci_n5k_e50_s*.redo_*.jsonl")):
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

by_label = {}
for r in rows:
    by_label.setdefault(r.get("label","?"), []).append(r)

print(f"\n=== aggregate over {len(rows)} rows from {len(set((r['label'], r['seed']) for r in rows))} (label, seed) cells ===")
for label, items in sorted(by_label.items()):
    aucs = [r.get("mAP_50") for r in items if "mAP_50" in r]
    box  = [r.get("box_cls_acc") for r in items if r.get("box_cls_acc") is not None]
    n = len(items)
    if aucs:
        print(f"  {label:<18s}  n={n}  mAP50 mean={statistics.mean(aucs):.4f}  pstdev={statistics.pstdev(aucs):.4f}  "
              f"box_acc mean={statistics.mean(box):.3f}")
    else:
        print(f"  {label:<18s}  n={n}  (no mAP50 fields)")
PY

echo "[$(date -Is)] === redo DONE ===" | tee -a "$MASTER"
