#!/usr/bin/env bash
# Stage D-3-BREAK Phase 9 — 5-seed validation of B9 (320 px, ep=180).
#
# Single-seed B9 (Phase 8, this evening) hit mAP_50 = 0.1213 with
# cls_acc = 0.556 — +4σ over C9 5-seed (0.0790 ± 0.0105), the FIRST
# 320 px cell to clear C9.  Phase 9 promotes that to a 5-seed claim
# so it can land in the paper's Table I rather than future-work prose.
#
# Five seeds with the same config as B9 (320 px, b=8, ep=180, nodelet,
# λ_gate=2.0, lr=3e-3, resnet18_imagenet).  Falsifier:
#
#   mean ≥ C9 + 1σ AND all 5 seeds individually > C9 mean
#       ⇒ confirmed publishable lift over C9
#   mean within C9 band [0.0685, 0.0895]
#       ⇒ B9 was a lucky seed; ladder closes here
#   any seed OOM
#       ⇒ unexpected; investigate before any further runs
#
# Expected wall per seed ≈ 5345 s = 89 min.  Total ≈ 7.4 h.  Overnight-
# safe with 16 GiB cgroup MemoryMax and queue-behind sentinel.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/voc_b9_5seed_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/orchestrator.log"
GRID_JSONL="${OUT_DIR}/grid.jsonl"
: > "$GRID_JSONL"

echo "=== Stage D-3-BREAK Phase 9 — B9 5-seed validation ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind in-flight VOC / Gömb training.
echo "[orch] waiting for GPU..." | tee -a "$LOG"
while pgrep -af 'train_voc_stagec|run_gomb_smoke|ros2 launch hymeko_ros2_demo' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

run_seed () {
  local seed="$1"
  local cell="B9s${seed}_320_ep180"
  local logf="${OUT_DIR}/${cell}.log"
  local jsonl="${OUT_DIR}/${cell}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] CELL ${cell} START seed=${seed}" | tee -a "$LOG"

  systemd-run --user --scope --quiet \
    --unit="voc_b9_5seed_${STAMP}_${cell}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    python -m signedkan_wip.src.vision.train_voc_stagec \
      --image-set trainval --epochs 180 --input-size 320 \
      --batch-size 8 --n-box-queries 6 \
      --lr 0.003 --seed "$seed" \
      --device cuda \
      --backbone resnet18_imagenet \
      --query-head-kind nodelet \
      --lam-gate-neg 2.0 \
      --gate-loss-kind bce \
      --jsonl-out "$jsonl" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))

  if [ -s "$jsonl" ]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("$jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
out = pathlib.Path("$GRID_JSONL")
with out.open("a") as f:
    for r in rows:
        r["_cell"] = "$cell"
        r["_seed"] = $seed
        f.write(json.dumps(r) + "\n")
PY
  fi

  local m
  m=$(python -c "
import json, pathlib
try:
    rows = [json.loads(l) for l in pathlib.Path('$jsonl').read_text().splitlines() if l.strip()]
    last = rows[-1] if rows else {}
    m = last.get('mAP_50')
    print(f'{m:.4f}' if isinstance(m, float) else str(m))
except Exception as exc:
    print(f'err:{exc}')
" 2>&1)
  echo "[$(date -Is)] CELL ${cell} DONE rc=$rc seed=$seed mAP_50=$m elapsed=${elapsed}s" | tee -a "$LOG"
}

# Five seeds.  Seed 0 = B9 itself (re-run for parity with the other 4).
for s in 0 1 2 3 4; do
  run_seed "$s"
done

# Aggregator
python - <<PY | tee -a "$LOG"
import json, math, pathlib
rows = [json.loads(l) for l in pathlib.Path("$GRID_JSONL").read_text().splitlines() if l.strip()]
rows = [r for r in rows if isinstance(r.get("mAP_50"), float)]

C9_MEAN = 0.0790
C9_SD   = 0.0105
B9_SINGLE = 0.1213
N_C9 = 5

vals = [r["mAP_50"] for r in rows]
n = len(vals)
mean = sum(vals) / n if n else float("nan")
var  = sum((v - mean) ** 2 for v in vals) / n if n else float("nan")
sd   = math.sqrt(var) if n else float("nan")

print()
print(f"=== Phase 9 B9 5-seed validation ===")
print(f"{'cell':24s} {'seed':>4s} {'mAP':>7s} {'mIoU':>6s} {'cls_acc':>8s} {'wall_s':>8s}")
for r in sorted(rows, key=lambda r: r['_seed']):
    print(f"{r['_cell']:24s} {r['_seed']:>4d} {r['mAP_50']:>7.4f} "
          f"{r.get('mean_iou_matched', float('nan')):>6.3f} "
          f"{r.get('box_cls_acc', float('nan')):>8.3f} "
          f"{r.get('wall_s', float('nan')):>8.1f}")
print()
print(f"B9 5-seed:        mean = {mean:.4f}  pstdev = {sd:.4f}  n = {n}")
print(f"C9 5-seed anchor: mean = {C9_MEAN:.4f}  pstdev = {C9_SD:.4f}  n = {N_C9}")
print(f"Phase 8 single seed reference: {B9_SINGLE:.4f}")
print()

# Paired-like t-stat against C9 (Welch, treating C9 as fixed-pop):
if n > 1 and sd > 0:
    sigma_diff = math.sqrt(sd**2 / n + C9_SD**2 / N_C9)
    t = (mean - C9_MEAN) / sigma_diff if sigma_diff > 0 else float('nan')
    print(f"Welch-like t vs C9 anchor: t = {t:+.2f}  (n_B9={n}, n_C9={N_C9})")

n_in_band = sum(1 for v in vals if C9_MEAN - C9_SD <= v <= C9_MEAN + C9_SD)
n_above   = sum(1 for v in vals if v > C9_MEAN + C9_SD)
n_below   = sum(1 for v in vals if v < C9_MEAN - C9_SD)
print(f"Win-rate vs C9: {n_above}/{n} above C9 band, "
      f"{n_in_band}/{n} in band, {n_below}/{n} below band")

if mean > C9_MEAN + C9_SD and n_above == n:
    print("VERDICT: ✓ confirmed publishable lift over C9 (5/5 above band, mean > C9+σ)")
elif C9_MEAN - C9_SD <= mean <= C9_MEAN + C9_SD:
    print("VERDICT: · 5-seed mean within C9 band — B9 single-seed was a lucky draw")
else:
    print("VERDICT: ? mixed — n_above/n_in/n_below split; inspect per-seed details")
PY

for cl in B9s0 B9s1 B9s2 B9s3 B9s4; do
  log=$(ls $OUT_DIR/${cl}_*.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  if grep -q 'CUDA out of memory' "$log" 2>/dev/null; then
    echo "[orch] $(basename $log .log) hit OOM" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== Phase 9 DONE $(date -Is) ===" | tee -a "$LOG"
echo "  grid_jsonl: $GRID_JSONL" | tee -a "$LOG"
