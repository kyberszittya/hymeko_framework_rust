#!/usr/bin/env bash
# Sequential HSiKAN — first real-text benchmark (IMDB binary sentiment).
# Plan: docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/.
#
# Two-stage launch per CLAUDE §3:
#   1. PRODUCTION-SCALE SMOKE (1 seed × full IMDB × 5 epochs).
#      Falsifier gate: val_acc < 0.65 blocks the 5-seed launch.
#      Budget: ~15 min GPU.
#
#   2. 5-SEED HEADLINE (5 seeds × full IMDB × 20 epochs).
#      Falsifier: 5-seed mean test_accuracy < 0.70 → architecture
#      does not transfer to natural language at this scale.
#      Budget: ~2 h GPU.
#
# Queues behind any in-flight signedkan_wip training (HyMeYOLO
# b_hsikan rerun, run_final_cell, etc.).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/seq_hsikan_imdb_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
SMOKE_JSONL="${OUT_DIR}/smoke_seed0.jsonl"
FIVE_SEED_JSONL="${OUT_DIR}/5seed.jsonl"

echo "=== Sequential HSiKAN IMDB ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight signedkan_wip training.
echo "[orch] waiting for GPU..." | tee -a "$LOG"
while pgrep -af 'signedkan_wip\.src\.(run_optuna_search|run_final_cell|run_gomb_smoke|vision\.train_circles_ricci|vision\.train_voc)' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

# ─── Stage 1: production-scale smoke ──────────────────────────────────

echo "" | tee -a "$LOG"
echo "## Stage 1 — production-scale smoke (1 seed × full IMDB × 5 epochs)" | tee -a "$LOG"

SMOKE_LOG="${OUT_DIR}/smoke_seed0.log"
t0=$(date +%s)
echo "[$(date -Is)] SMOKE START seed=0" | tee -a "$LOG"

systemd-run --user --scope --quiet \
  --unit="seq_hsikan_imdb_smoke_${STAMP}.scope" \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  python -u -m signedkan_wip.src.sequence.train_imdb_classifier \
    --epochs 5 --batch-size 32 --lr 3e-4 --seed 0 \
    --device cuda \
    --jsonl-out "$SMOKE_JSONL" \
  > "$SMOKE_LOG" 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
echo "[$(date -Is)] SMOKE DONE rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"

SMOKE_VAL=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$SMOKE_JSONL').read_text().splitlines() if l.strip()]
    if rows:
        v = rows[-1].get('best_val_accuracy')
        print(f'{v:.4f}' if isinstance(v, (int, float)) else 'null')
    else:
        print('empty')
except FileNotFoundError:
    print('missing')
" 2>&1)
echo "[orch] smoke val_acc=$SMOKE_VAL" | tee -a "$LOG"

# Gate: val_acc must be >= 0.65 to queue the 5-seed.
if [ "$SMOKE_VAL" = "null" ] || [ "$SMOKE_VAL" = "empty" ] || [ "$SMOKE_VAL" = "missing" ]; then
  echo "[orch] SMOKE PRODUCED NO val_acc — aborting" | tee -a "$LOG"
  exit 1
fi
below_floor=$(python -c "print(1 if float('$SMOKE_VAL') < 0.65 else 0)" 2>&1)
if [ "$below_floor" = "1" ]; then
  echo "[orch] SMOKE val_acc=$SMOKE_VAL < 0.65 — aborting 5-seed launch (plan §5 gate)" | tee -a "$LOG"
  exit 2
fi
echo "[orch] SMOKE PASSES gate (val_acc=$SMOKE_VAL >= 0.65), proceeding to 5-seed" | tee -a "$LOG"

# ─── Stage 2: 5-seed headline run ────────────────────────────────────

echo "" | tee -a "$LOG"
echo "## Stage 2 — 5-seed headline (5 × full IMDB × 20 epochs)" | tee -a "$LOG"
: > "$FIVE_SEED_JSONL"

run_one() {
  local seed="$1"
  local logf="${OUT_DIR}/5seed_seed${seed}.log"
  local jsonl="${OUT_DIR}/5seed_seed${seed}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] 5SEED START seed=$seed" | tee -a "$LOG"
  systemd-run --user --scope --quiet \
    --unit="seq_hsikan_imdb_5seed_${STAMP}_s${seed}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -u -m signedkan_wip.src.sequence.train_imdb_classifier \
      --epochs 20 --batch-size 32 --lr 3e-4 --seed "$seed" \
      --device cuda \
      --jsonl-out "$jsonl" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ -s "$jsonl" ]; then
    cat "$jsonl" >> "$FIVE_SEED_JSONL"
  fi
  local acc
  acc=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$jsonl').read_text().splitlines() if l.strip()]
    v = rows[-1].get('test_accuracy') if rows else None
    print(f'{v:.4f}' if isinstance(v, (int, float)) else 'null')
except Exception:
    print('error')
" 2>&1)
  echo "[$(date -Is)] 5SEED DONE  seed=$seed rc=$rc test_acc=$acc elapsed=${elapsed}s" | tee -a "$LOG"
}

for SEED in 0 1 2 3 4; do
  run_one "$SEED"
done

# Aggregate
python - <<PY | tee -a "$LOG"
import json, statistics, pathlib
rows = [json.loads(l) for l in pathlib.Path("$FIVE_SEED_JSONL").read_text().splitlines() if l.strip()]
accs = [r["test_accuracy"] for r in rows if isinstance(r.get("test_accuracy"), (int, float))]
print(f"[5-seed aggregate] n={len(accs)}")
if accs:
    mean = statistics.mean(accs)
    sd = statistics.pstdev(accs) if len(accs) >= 2 else 0.0
    print(f"  test_accuracy mean = {mean:.4f}  pstdev = {sd:.4f}")
    print(f"  per-seed: {[f'{a:.4f}' for a in accs]}")
    if mean < 0.70:
        print(f"  FALSIFIER HIT: 5-seed mean < 0.70 — plan §6 rollback path active")
    elif mean < 0.80:
        print(f"  RECIPE-NEEDS-WORK zone (0.70 <= mean < 0.80) — open Stage 2")
    else:
        print(f"  PASS: mean >= 0.80")
PY

echo "" | tee -a "$LOG"
echo "=== Sequential HSiKAN IMDB DONE $(date -Is) ===" | tee -a "$LOG"
echo "  smoke jsonl: $SMOKE_JSONL" | tee -a "$LOG"
echo "  5-seed jsonl: $FIVE_SEED_JSONL" | tee -a "$LOG"
