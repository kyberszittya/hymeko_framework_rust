#!/usr/bin/env bash
# IMDB architectural-fairness experiment.
#
# Per the 2026-05-18 follow-up to the Sequential HSiKAN IMDB result:
# the May-17 5-seed (0.8395 ± 0.0058) compared HSiKAN-from-scratch to
# a pretrained-BERT mental baseline, which is an architecturally
# unfair comparison (BERT had ~3 B tokens of pretraining; we had 25 k
# labeled).
#
# This experiment fixes the comparison:
#
#   Phase 1 — HSiKAN MLM pretrain on IMDB unsup 50k (~10–15 M tokens).
#   Phase 2 — Transformer MLM pretrain on IMDB unsup 50k (iso-corpus).
#   Phase 3 — HSiKAN pretrained → fine-tune labeled 25k, 5 seeds.
#   Phase 4 — Transformer FROM-SCRATCH 5 seeds (architectural fairness
#             baseline).
#   Phase 5 — Transformer pretrained → fine-tune labeled 25k, 5 seeds.
#
# The May-17 HSiKAN from-scratch 5-seed (0.8395) is reused as the
# reference HSiKAN-from-scratch number; it used the same vocab
# (the orchestrator pins UNK_ID as the MLM mask sentinel to avoid
# vocab-drift between from-scratch and pretrain runs).
#
# Queues behind any in-flight signedkan_wip training.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="signedkan_wip/experiments/results/imdb_arch_fairness_${STAMP}"
mkdir -p "$OUT_DIR" "$OUT_DIR/pretrain_ckpts"
LOG="${OUT_DIR}/orchestrator.log"
PHASE3_JSONL="${OUT_DIR}/phase3_hsikan_pretrained.jsonl"
PHASE4_JSONL="${OUT_DIR}/phase4_transformer_from_scratch.jsonl"
PHASE5_JSONL="${OUT_DIR}/phase5_transformer_pretrained.jsonl"
: > "$PHASE3_JSONL"
: > "$PHASE4_JSONL"
: > "$PHASE5_JSONL"

echo "=== IMDB Architectural-Fairness Experiment ===" | tee -a "$LOG"
echo "stamp=$STAMP" | tee -a "$LOG"
echo "out_dir=$OUT_DIR" | tee -a "$LOG"
echo "git SHA: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

# Queue behind any in-flight signedkan_wip training.
echo "[orch] waiting for GPU..." | tee -a "$LOG"
while pgrep -af 'signedkan_wip\.src\.(run_optuna_search|run_final_cell|run_gomb_smoke|vision\.train_circles_ricci|vision\.train_voc|sequence\.(train|run))' \
        | grep -v "$$" | grep -v "$0" | grep -q .; do
  sleep 30
done
echo "[orch] GPU free, starting $(date -Is)" | tee -a "$LOG"

HSIKAN_PRETRAIN_PT="${OUT_DIR}/pretrain_ckpts/hsikan_unsup.pt"
HSIKAN_PRETRAIN_JSONL="${OUT_DIR}/phase1_hsikan_pretrain.jsonl"
XFM_PRETRAIN_PT="${OUT_DIR}/pretrain_ckpts/transformer_unsup.pt"
XFM_PRETRAIN_JSONL="${OUT_DIR}/phase2_transformer_pretrain.jsonl"

# ─── Phase 1 — HSiKAN MLM pretrain on unsup 50k ──────────────────────

echo "" | tee -a "$LOG"
echo "## Phase 1 — HSiKAN MLM pretrain (unsup 50k × 20 epochs)" | tee -a "$LOG"
t0=$(date +%s)
systemd-run --user --scope --quiet \
  --unit="imdb_arch_fair_p1_${STAMP}.scope" \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  python -u -m signedkan_wip.src.sequence.run_imdb_mlm_pretrain \
    --arch hsikan --epochs 20 --batch-size 32 --lr 5e-4 --seed 0 \
    --device cuda \
    --state-dict-out "$HSIKAN_PRETRAIN_PT" \
    --jsonl-out "$HSIKAN_PRETRAIN_JSONL" \
  > "${OUT_DIR}/phase1_hsikan_pretrain.log" 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
echo "[$(date -Is)] Phase 1 DONE rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"

# ─── Phase 2 — Transformer MLM pretrain on unsup 50k ─────────────────

echo "" | tee -a "$LOG"
echo "## Phase 2 — Transformer MLM pretrain (unsup 50k × 20 epochs)" | tee -a "$LOG"
t0=$(date +%s)
systemd-run --user --scope --quiet \
  --unit="imdb_arch_fair_p2_${STAMP}.scope" \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  python -u -m signedkan_wip.src.sequence.run_imdb_mlm_pretrain \
    --arch transformer --epochs 20 --batch-size 32 --lr 5e-4 --seed 0 \
    --device cuda \
    --state-dict-out "$XFM_PRETRAIN_PT" \
    --jsonl-out "$XFM_PRETRAIN_JSONL" \
  > "${OUT_DIR}/phase2_transformer_pretrain.log" 2>&1
rc=$?
elapsed=$(( $(date +%s) - t0 ))
echo "[$(date -Is)] Phase 2 DONE rc=$rc elapsed=${elapsed}s" | tee -a "$LOG"

# Helper for fine-tune 5-seed runs.
run_finetune() {
  local phase="$1"; shift
  local arch="$1"; shift   # hsikan / transformer
  local pretrained="$1"; shift  # path to .pt or "" for from-scratch
  local jsonl_acc="$1"; shift
  local seed="$1"; shift
  local logf="${OUT_DIR}/${phase}_seed${seed}.log"
  local jsonl="${OUT_DIR}/${phase}_seed${seed}.jsonl"
  local t0; t0=$(date +%s)
  echo "[$(date -Is)] $phase START seed=$seed" | tee -a "$LOG"

  local cmd=("python" "-u" "-m" "signedkan_wip.src.sequence.train_imdb_classifier")
  if [ "$arch" = "transformer" ]; then
    cmd=("python" "-u" "-m" "signedkan_wip.src.sequence.train_imdb_transformer")
  fi
  local extra=()
  if [ -n "$pretrained" ]; then
    extra+=("--pretrained-state-dict" "$pretrained")
  fi

  systemd-run --user --scope --quiet \
    --unit="imdb_arch_fair_${phase}_${STAMP}_s${seed}.scope" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    "${cmd[@]}" \
      --epochs 20 --batch-size 32 --lr 3e-4 --seed "$seed" \
      --device cuda \
      --jsonl-out "$jsonl" \
      "${extra[@]}" \
    > "$logf" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ -s "$jsonl" ]; then
    cat "$jsonl" >> "$jsonl_acc"
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
  echo "[$(date -Is)] $phase DONE  seed=$seed rc=$rc test_acc=$acc elapsed=${elapsed}s" \
    | tee -a "$LOG"
}

# ─── Phase 3 — HSiKAN pretrained → fine-tune 5-seed ──────────────────

echo "" | tee -a "$LOG"
echo "## Phase 3 — HSiKAN pretrained → fine-tune (5-seed × 20 epochs)" | tee -a "$LOG"
for SEED in 0 1 2 3 4; do
  run_finetune "phase3_hsikan_pretrained" "hsikan" "$HSIKAN_PRETRAIN_PT" \
               "$PHASE3_JSONL" "$SEED"
done

# ─── Phase 4 — Transformer FROM SCRATCH 5-seed (arch-fairness) ──────

echo "" | tee -a "$LOG"
echo "## Phase 4 — Transformer FROM-SCRATCH (5-seed × 20 epochs) — arch-fairness" | tee -a "$LOG"
for SEED in 0 1 2 3 4; do
  run_finetune "phase4_transformer_from_scratch" "transformer" "" \
               "$PHASE4_JSONL" "$SEED"
done

# ─── Phase 5 — Transformer pretrained → fine-tune 5-seed ────────────

echo "" | tee -a "$LOG"
echo "## Phase 5 — Transformer pretrained → fine-tune (5-seed × 20 epochs)" | tee -a "$LOG"
for SEED in 0 1 2 3 4; do
  run_finetune "phase5_transformer_pretrained" "transformer" "$XFM_PRETRAIN_PT" \
               "$PHASE5_JSONL" "$SEED"
done

# ─── Aggregate ─────────────────────────────────────────────────────

python - <<PY | tee -a "$LOG"
import json, statistics, pathlib
def agg(p):
    rows = [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
    accs = [r["test_accuracy"] for r in rows if isinstance(r.get("test_accuracy"), (int, float))]
    return accs

print("")
print("## Architectural-fairness summary")
print("")
print("                                                  n    mean    pstdev   seeds")
for phase, path in [
    ("Phase 3 HSiKAN pretrained",       "$PHASE3_JSONL"),
    ("Phase 4 Transformer from-scratch","$PHASE4_JSONL"),
    ("Phase 5 Transformer pretrained",  "$PHASE5_JSONL"),
]:
    accs = agg(path)
    if accs:
        mean = statistics.mean(accs)
        sd = statistics.pstdev(accs) if len(accs) >= 2 else 0.0
        seeds_str = ', '.join(f'{a:.4f}' for a in accs)
        print(f"  {phase:<48s} {len(accs):>2d}   {mean:.4f}   {sd:.4f}   [{seeds_str}]")
    else:
        print(f"  {phase:<48s} (no data)")
print("")
print("Reference: HSiKAN from-scratch (May 17) = 0.8395 ± 0.0058 (5 seeds)")
PY

echo "" | tee -a "$LOG"
echo "=== IMDB Arch-Fairness DONE $(date -Is) ===" | tee -a "$LOG"
echo "  jsonls:" | tee -a "$LOG"
echo "    phase 3: $PHASE3_JSONL" | tee -a "$LOG"
echo "    phase 4: $PHASE4_JSONL" | tee -a "$LOG"
echo "    phase 5: $PHASE5_JSONL" | tee -a "$LOG"
