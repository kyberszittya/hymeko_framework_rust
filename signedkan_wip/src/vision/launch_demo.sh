#!/usr/bin/env bash
# HyMeYOLO GUI launcher — auto-detect or train a checkpoint, then open Tk.
#
# Usage:
#   bash signedkan_wip/src/vision/launch_demo.sh             # auto: latest CMNIST ckpt or quick-train
#   bash signedkan_wip/src/vision/launch_demo.sh a2          # train Stage A-2 then launch
#   bash signedkan_wip/src/vision/launch_demo.sh b_resnet    # train Stage B then launch
#   bash signedkan_wip/src/vision/launch_demo.sh b_hsikan    # train Stage B' (HSiKAN-CR) then launch
#   bash signedkan_wip/src/vision/launch_demo.sh c_fpn       # train Stage C then launch
#   bash signedkan_wip/src/vision/launch_demo.sh quick       # force quick-train (no save)
#   bash signedkan_wip/src/vision/launch_demo.sh list        # list known ckpts and exit
#
# Stage configs match reports/2026-05-17-hymeyolo-stage-c-5seed.md.
# Each STAGE training takes ~10-15 min on GPU; quick-train is ~50 s on CPU.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PATH="/home/kyberszittya/miniconda3/bin:$PATH"

CKPT_ROOT="${HYMEYOLO_CKPT_ROOT:-$REPO_ROOT/checkpoints/hymeyolo_demo}"
mkdir -p "$CKPT_ROOT"

ACTION="${1:-auto}"

# Stage configurations — canonical flag sets matching
# run_hymeyolo_ladder_5seed.sh (the published-result configs).
# b_resnet / b_hsikan / c_fpn add the A-3-lite bundle (LayerNorm,
# weight decay 1e-4, focal classification loss) — these are the
# flags that produced the 0.8955 / 0.9032 / 0.8926 5-seed means.
declare -A STAGE_FLAGS=(
  [a2]="--warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01"
  [b_resnet]="--warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 --use-layernorm --weight-decay 1e-4 --cls-loss focal --backbone resnet"
  [b_hsikan]="--warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 --use-layernorm --weight-decay 1e-4 --cls-loss focal --backbone hsikan"
  [c_fpn]="--warm-start --schedule cosine --warmup-epochs 10 --min-lr-ratio 0.01 --use-layernorm --weight-decay 1e-4 --cls-loss focal --backbone resnet --fpn 2level"
)

# Per-stage training defaults.
N_IMAGES_DEFAULT=5000
EPOCHS_DEFAULT=100
LR_DEFAULT=0.003

list_ckpts() {
  echo "CMNIST checkpoints (load-able by demo_hymeyolo_tk):"
  _cmnist_ckpt_rows \
    | sort -rn \
    | awk '{printf "  %s  ", strftime("%Y-%m-%d %H:%M", $1); $1=""; sub(/^ +/, ""); print}' \
    | head -20
  echo
  echo "(searched: $CKPT_ROOT,"
  echo "  $REPO_ROOT/checkpoints/hymeyolo, /tmp/hymeyolo_*_ckpts,"
  echo "  signedkan_wip/experiments/results/*/ckpts/)"
  echo
  echo "Other on-disk checkpoints (NOT load-able — different schema):"
  find "$REPO_ROOT/signedkan_wip/experiments/results" \
      -maxdepth 4 -name 'stage_c_voc*.pt' \
      -printf "  %TY-%Tm-%Td %TH:%TM  %p\n" 2>/dev/null | sort -r
}

# Internal helper — return all candidate ckpt rows (mtime + path), unsorted.
# Searches the demo ckpt root, /tmp scratch dirs, and ladder-run ckpt
# subdirs under signedkan_wip/experiments/results/. Excludes VOC ckpts
# (which the demo can't load — different n_classes / input_size).
_cmnist_ckpt_rows() {
  for d in \
      "$CKPT_ROOT" \
      "$REPO_ROOT/checkpoints/hymeyolo" \
      "/tmp/hymeyolo_ckpts" \
      "/tmp/hymeyolo_demo_ckpts"; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 3 -name "*.pt" -not -name 'stage_c_voc*' \
      -printf "%T@ %p\n" 2>/dev/null
  done
  # Ladder-run ckpts under results/<run>/ckpts/ (added 2026-05-18 —
  # ladder script now passes --save-checkpoint).
  find "$REPO_ROOT/signedkan_wip/experiments/results" \
      -maxdepth 4 -path '*/ckpts/ricci-mod_seed*.pt' \
      -not -path '*stage_d_voc*' \
      -printf "%T@ %p\n" 2>/dev/null
}

latest_cmnist_ckpt() {
  _cmnist_ckpt_rows | sort -rn | head -1 | awk '{print $2}'
}

train_stage() {
  local stage="$1"
  local flags="${STAGE_FLAGS[$stage]:-}"
  if [ -z "$flags" ]; then
    echo "Unknown stage: '$stage'. Known: ${!STAGE_FLAGS[*]}"
    exit 2
  fi
  local out_dir="$CKPT_ROOT/$stage"
  mkdir -p "$out_dir"
  echo "[launcher] training Stage '$stage' → $out_dir"
  echo "[launcher] flags: $flags"
  python -m signedkan_wip.src.vision.train_circles_ricci \
    --n-images "$N_IMAGES_DEFAULT" --epochs "$EPOCHS_DEFAULT" \
    --lr "$LR_DEFAULT" --ricci-scale 1.0 \
    $flags \
    --configs '+ricci-mod' \
    --seed 0 \
    --save-checkpoint "$out_dir" \
    --jsonl-out "$out_dir/train.jsonl"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[launcher] training failed (rc=$rc); aborting"
    exit $rc
  fi
  local ckpt
  ckpt=$(find "$out_dir" -maxdepth 1 -name "ricci-mod_seed0.pt" -print -quit)
  if [ -z "$ckpt" ]; then
    echo "[launcher] training finished but no ckpt found in $out_dir"
    exit 3
  fi
  echo "$ckpt"
}

case "$ACTION" in
  list)
    list_ckpts
    exit 0
    ;;
  quick)
    echo "[launcher] force quick-train (no save)"
    python -m signedkan_wip.src.vision.demo_hymeyolo_tk
    ;;
  auto)
    ckpt=$(latest_cmnist_ckpt)
    if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
      echo "[launcher] auto-detected ckpt: $ckpt"
      python -m signedkan_wip.src.vision.demo_hymeyolo_tk --checkpoint "$ckpt"
    else
      echo "[launcher] no CMNIST ckpt found — quick-training a small model"
      echo "[launcher] (run with 'a2' / 'b_resnet' / 'b_hsikan' / 'c_fpn' for a real ckpt)"
      python -m signedkan_wip.src.vision.demo_hymeyolo_tk
    fi
    ;;
  a2|b_resnet|b_hsikan|c_fpn)
    ckpt=$(train_stage "$ACTION" | tail -1)
    if [ -f "$ckpt" ]; then
      echo "[launcher] launching demo against $ckpt"
      python -m signedkan_wip.src.vision.demo_hymeyolo_tk --checkpoint "$ckpt"
    else
      echo "[launcher] could not locate trained ckpt; falling back to quick-train"
      python -m signedkan_wip.src.vision.demo_hymeyolo_tk
    fi
    ;;
  *)
    echo "Unknown action '$ACTION'."
    echo "Usage: $0 [auto|quick|list|a2|b_resnet|b_hsikan|c_fpn]"
    exit 2
    ;;
esac
