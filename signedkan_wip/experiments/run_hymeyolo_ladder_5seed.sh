#!/usr/bin/env bash
#
# Unified HyMeYOLO YOLO-parity-ladder 5-seed orchestrator.
#
# Replaces per-stage `.sh` scripts. Pick a stage name; the orchestrator
# constructs the canonical CLI for that stage and runs 5 seeds under
# the cgroup cap. Same per-seed dataset realisation across all stages
# (deterministic from --seed); per-stage results land under
# signedkan_wip/experiments/results/hymeyolo_ladder_<stage>_<STAMP>/
# so the analyser can paired-compare adjacent stages.
#
# Usage:
#     ./run_hymeyolo_ladder_5seed.sh <stage>
#
# Stages (canonical names from the 2026-05-16 YOLO-parity ladder):
#     baseline   honest, no-warm-start, const LR, e=50 (the pre-A1 control)
#     a1         + warm-start
#     a2         + cosine LR + warmup + e=100
#     a3         + LayerNorm + WeightDecay + focal cls + GIoU box
#     b          + ResNet-tiny backbone swap (NOT YET implemented)
#     c          + FPN multi-scale heads (NOT YET implemented)
#
# Environment overrides:
#     N_IMAGES    (default 5000)
#     SEEDS       (default "0 1 2 3 4")
#     LR          (default 0.003)
#
# Plan / report convention:
#     docs/plans/2026-05-16-hymeyolo-stage-<stage>-*/
#     reports/2026-05-16-hymeyolo-stage-<stage>-5seed.md

set -euo pipefail

REPO_ROOT="/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust"
cd "$REPO_ROOT"

export PATH="/home/kyberszittya/miniconda3/bin:$PATH"
PY=/home/kyberszittya/miniconda3/bin/python

STAGE="${1:?usage: $0 <stage>}"
N_IMAGES="${N_IMAGES:-5000}"
LR="${LR:-0.003}"
SEEDS_LIST="${SEEDS:-0 1 2 3 4}"

# Per-stage CLI fragments. Each fragment is a bash array of CLI args
# appended to the base train_circles_ricci command.
declare -a STAGE_FLAGS=()
declare -a STAGE_EPOCHS=()

case "$STAGE" in
  baseline)
    STAGE_FLAGS+=(--no-warm-start --schedule constant --warmup-epochs 0)
    EPOCHS=50
    PER_RUN_TIMEOUT=1800
    ;;
  a1)
    STAGE_FLAGS+=(--warm-start --schedule constant --warmup-epochs 0)
    EPOCHS=50
    PER_RUN_TIMEOUT=1800
    ;;
  a2)
    STAGE_FLAGS+=(--warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01)
    EPOCHS=100
    PER_RUN_TIMEOUT=2400
    ;;
  a3)
    # Full Stage A-3: 4 sub-levers. GIoU's per-image torch.stack
    # pattern in `_box_loss_on_matched` causes ~2.3× per-epoch
    # slowdown on the RTX 2070 SUPER (smoke hit 2400s timeout
    # mid-training 2026-05-16). Use a3 only after the GIoU branch
    # is vectorised, or budget a longer per-run timeout.
    STAGE_FLAGS+=(
      --warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01
      --use-layernorm --weight-decay 1e-4
      --cls-loss focal --box-loss giou
    )
    EPOCHS=100
    PER_RUN_TIMEOUT=3600
    ;;
  a3_lite)
    # Stage A-3 minus the GIoU lever — the conservative 3-lever
    # variant that ships at Stage A-2's wall budget. Keeps
    # LayerNorm + weight_decay + focal cls; box loss stays on L1.
    # Used 2026-05-16 evening when the full a3 smoke timed out;
    # paired-compares cleanly against a2 since the only delta is
    # the 3 levers turned on.
    STAGE_FLAGS+=(
      --warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01
      --use-layernorm --weight-decay 1e-4
      --cls-loss focal
    )
    EPOCHS=100
    PER_RUN_TIMEOUT=2400
    ;;
  b_resnet)
    # Stage B: ResNet-tiny backbone (deeper, residual). Includes the
    # Stage A-3-lite levers (LayerNorm + WD + focal) since the smoke
    # showed they're effectively free at this protocol scale.
    STAGE_FLAGS+=(
      --warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01
      --use-layernorm --weight-decay 1e-4 --cls-loss focal
      --backbone resnet
    )
    EPOCHS=100
    PER_RUN_TIMEOUT=3000
    ;;
  b_hsikan)
    # Stage B': same architecture as b_resnet but Catmull-Rom basis-
    # function activations replace every ReLU. Isolates whether the
    # HSiKAN family's basis-function primitive transfers to vision
    # convolutional features (the σ-cycle aggregator does NOT, per
    # the 2026-05-14 vision-corner negative and 2026-05-16 tabular
    # sanity). Paired comparison vs b_resnet at iso-topology.
    STAGE_FLAGS+=(
      --warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01
      --use-layernorm --weight-decay 1e-4 --cls-loss focal
      --backbone hsikan
    )
    EPOCHS=100
    PER_RUN_TIMEOUT=3000
    ;;
  b_prime)
    # Stage B' ATTRIBUTION CONTROL: TinyBackbone + A-3-lite levers.
    # Same as Stage A-3-lite but recognised as its own ladder step
    # so the orchestrator + paired analyser address it by name.
    # Paired vs Stage A-2 isolates the A-3-lite bundle's lift;
    # paired vs Stage B b_resnet isolates the backbone-swap lift.
    # See docs/plans/2026-05-16-hymeyolo-stage-b-prime-control/.
    #
    # 2026-05-17: PER_RUN_TIMEOUT bumped 2400 -> 3600. The original
    # 2400s ran cleanly under sole-GPU, but the overnight
    # 2026-05-16 attempt SIGKILL'd 5/5 seeds under concurrent-GPU
    # contention with b_hsikan reruns. Generous cushion eliminates
    # the timeout-at-the-edge risk if any future contention recurs.
    STAGE_FLAGS+=(
      --warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01
      --use-layernorm --weight-decay 1e-4 --cls-loss focal
      --backbone tiny
    )
    EPOCHS=100
    PER_RUN_TIMEOUT=3600
    ;;
  c_fpn)
    # Stage C: 2-level FPN multi-scale heads (P2 at /4 + P3 at /8)
    # on top of the ResNet-tiny backbone. Multi-scale bilinear
    # sampling at query corners; 1x1 lateral + top-down upsample
    # + 3x3 smooth. ~11k FPN params on top of the 107k backbone.
    # See docs/plans/2026-05-16-hymeyolo-stage-c-fpn/.
    # Predicted +0.03 paired vs Stage B b_resnet.
    STAGE_FLAGS+=(
      --warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01
      --use-layernorm --weight-decay 1e-4 --cls-loss focal
      --backbone resnet --fpn 2level
    )
    EPOCHS=100
    PER_RUN_TIMEOUT=3600
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Known stages: baseline a1 a2 a3_lite a3 b_resnet b_hsikan b_prime c_fpn" >&2
    exit 2
    ;;
esac

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/signedkan_wip/experiments/results/hymeyolo_ladder_${STAGE}_${STAMP}"
mkdir -p "$OUT_DIR"
MASTER="$OUT_DIR/orchestrator.log"

GIT_SHA="$(git rev-parse HEAD)"
echo "[$(date -Is)] ladder stage=$STAGE start  STAMP=$STAMP  git=$GIT_SHA" \
  | tee -a "$MASTER"
echo "[$(date -Is)] stage flags: ${STAGE_FLAGS[*]} epochs=$EPOCHS" \
  | tee -a "$MASTER"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  2>&1 | tee -a "$MASTER"

run_one() {
  local seed="$1"
  local slug="${STAGE}_seed${seed}_e${EPOCHS}"
  local jsonl_out="$OUT_DIR/${slug}.jsonl"
  local stdout_log="$OUT_DIR/${slug}.log"
  local stderr_log="$OUT_DIR/${slug}.err"
  local t0
  t0=$(date +%s)
  echo "[$(date -Is)] START seed=$seed" | tee -a "$MASTER"

  local ckpt_dir="$OUT_DIR/ckpts"
  local cmd=("$PY" -m signedkan_wip.src.vision.train_circles_ricci
             --n-images "$N_IMAGES" --epochs "$EPOCHS" --lr "$LR"
             --seed "$seed" --ricci-scale 1.0
             "${STAGE_FLAGS[@]}"
             --configs "+ricci-mod"
             --save-checkpoint "$ckpt_dir"
             --jsonl-out "$jsonl_out")

  local scope_name="hymeyolo-ladder-${slug}-${STAMP}.scope"
  systemd-run --user --scope --quiet \
    --unit="$scope_name" \
    -p MemoryMax=16G -p MemorySwapMax=0 \
    timeout "$PER_RUN_TIMEOUT" "${cmd[@]}" \
    > "$stdout_log" 2> "$stderr_log" || true
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [ -s "$jsonl_out" ]; then
    local row
    row=$(tail -1 "$jsonl_out")
    echo "[$(date -Is)] OK    seed=$seed rc=$rc elapsed=${elapsed}s row=$row" \
      | tee -a "$MASTER"
  else
    echo "[$(date -Is)] FAIL  seed=$seed rc=$rc elapsed=${elapsed}s NO_JSONL" \
      | tee -a "$MASTER"
  fi
}

for seed in $SEEDS_LIST; do
  run_one "$seed"
done

echo "[$(date -Is)] ladder stage=$STAGE end  $(ls -1 "$OUT_DIR"/*.jsonl 2>/dev/null | wc -l) rows" \
  | tee -a "$MASTER"
echo "Results: $OUT_DIR"
