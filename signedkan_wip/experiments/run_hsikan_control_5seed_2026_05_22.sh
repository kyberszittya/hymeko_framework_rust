#!/usr/bin/env bash
# HSIKAN control benchmark — 5-seed validation.
#
# Single-seed result (2026-05-21):
#   sinusoid  RMSE  HSIKAN=0.064  LQR=0.038  MPC=0.039  PP=0.350
#   s_curve   RMSE  HSIKAN=0.085  LQR=0.072  MPC=0.066  PP=0.160
#   straight  RMSE  HSIKAN=0.082  LQR=0.072  MPC=0.066  PP=0.134
# HSIKAN is within 1.5-2× of LQR/MPC, 22× faster than MPC on sinusoid.
#
# 5 seeds × 4 controllers × 3 tracks → 60 (controller,track,seed) cells.
# CPU-only — runs alongside GPU experiments without competing.
# Wall: ~5 min/seed × 5 = ~25 min total.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/hsikan_control_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
AGG_JSONL="${OUT_DIR}/all_seeds.jsonl"
: > "$AGG_JSONL"

echo "=== HSIKAN control 5-seed (CPU) ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

for SEED in 0 1 2 3 4; do
  jsonl="${OUT_DIR}/seed${SEED}.jsonl"
  logf="${OUT_DIR}/seed${SEED}.log"
  t0=$(date +%s)
  echo "[$(date -Is)] SEED ${SEED} START" | tee -a "$LOG"

  # 16 GiB cgroup cap (cheap insurance; CPU benchmark uses < 1 GiB).
  systemd-run --user --scope --quiet \
    --unit="hsikan_ctrl_${STAMP}_s${SEED}.scope" \
    -p MemoryMax=16G \
    python -m signedkan_wip.experiments.runs.run_control_benchmark_smoke \
      --seed "$SEED" --T 12 --train-epochs 200 --window 8 \
      --jsonl-out "$jsonl" \
    > "$logf" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))

  if [ -s "$jsonl" ]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
agg = pathlib.Path("$AGG_JSONL")
with agg.open("a") as f:
    for r in rows:
        r["_seed"] = $SEED
        f.write(json.dumps(r) + "\n")
PY
  fi
  echo "[$(date -Is)] SEED ${SEED} DONE rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"
done

# 5-seed aggregator across (controller × track)
python - <<PY | tee -a "$LOG"
import json, pathlib, statistics, collections
rows = [json.loads(l) for l in pathlib.Path("$AGG_JSONL").read_text().splitlines() if l.strip()]
buckets = collections.defaultdict(list)
for r in rows:
    key = (r["track"], r["controller"])
    buckets[key].append(r.get("lat_rmse"))

print()
print(f"=== HSIKAN control 5-seed (lateral RMSE in metres) ===")
print(f"{'track':14s} {'controller':14s} {'n':>3s} {'mean':>8s} {'pstdev':>8s} {'min':>8s} {'max':>8s}")
for (track, ctrl), vals in sorted(buckets.items()):
    vals = [v for v in vals if isinstance(v, float)]
    n = len(vals)
    if n == 0:
        continue
    mean = statistics.mean(vals)
    sd   = statistics.pstdev(vals) if n >= 2 else 0.0
    print(f"{track:14s} {ctrl:14s} {n:>3d} {mean:>8.4f} {sd:>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

# Paired Δ: HSIKAN vs each baseline per track
print()
print("=== Paired Δ (HSIKAN − baseline) per track ===")
for track in sorted({k[0] for k in buckets}):
    hk = sorted(buckets[(track, "hsikan")])
    for base_name in ("lqr", "pure_pursuit", "mpc"):
        bk = sorted(buckets.get((track, base_name), []))
        if len(hk) == len(bk) == 5:
            diffs = [h-b for h, b in zip(hk, bk)]
            mean = statistics.mean(diffs)
            sd = statistics.pstdev(diffs)
            z = mean / (sd / len(diffs) ** 0.5) if sd > 0 else float('inf')
            print(f"  {track:14s} hsikan-{base_name:14s}: Δmean={mean:+.4f}  σ_d={z:+.2f}")
PY

echo "" | tee -a "$LOG"
echo "=== DONE $(date -Is) ===" | tee -a "$LOG"
