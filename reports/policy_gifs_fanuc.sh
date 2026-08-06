#!/usr/bin/env bash
# Sequential: Galambos BC policies -> per-policy labeled GIFs -> FANUC BC->DDPG/TD3 (warm-start fix).
# Line-buffered (no log loss) + per-step SIGKILL timeout (no runaway). Best-effort: a failure logs and continues.
cd "d:/hakiko_ai_ws/03_implementation/hymeko_framework_rust" || exit 1
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=.
Q=reports/2026-06-24-policy-gifs-fanuc.log
mkdir -p checkpoints/galambos checkpoints/fanuc reports/gifs/policies docs/manipulation_models/results/gifs
: > "$Q"
run() { echo "" >> "$Q"; echo "=== $1 $(date +%H:%M:%S) ===" >> "$Q"; local to=$2; shift 2; stdbuf -oL timeout -s KILL "$to" "$@" >> "$Q" 2>&1 && echo "[ok]" >> "$Q" || echo "[FAILED rc=$?]" >> "$Q"; }

# 1. Galambos BC policies (BC only; robot=None to match the renderer)
run bc-hsikan 900 python -m hymeko_rl.galambos_bc --kind hsikan --algo ppo --refine 0 --n-demos 250 --bc-epochs 200 --difficulty 0.3 --save checkpoints/galambos/bc_ppo_hsikan.pt
run bc-mlp    900 python -m hymeko_rl.galambos_bc --kind mlp    --algo ppo --refine 0 --n-demos 250 --bc-epochs 200 --difficulty 0.3 --save checkpoints/galambos/bc_ppo_mlp.pt

# 2. per-policy GIFs on COMMON seeds (filename carries the policy + outcome)
SEEDS="0 9 10 14 1 2"
run gif-demonstrator 360 python -m hymeko_rl.render_planar_gifs --demonstrator --difficulty 0.3 --run policies
run gif-hsikan_bc    360 python -m hymeko_rl.render_planar_gifs --checkpoint checkpoints/galambos/bc_ppo_hsikan.pt --kind hsikan --algo ppo --hand-authored --label hsikan_bc --run policies --difficulty 0.3 --max-steps 300 --seeds $SEEDS
run gif-mlp_bc       360 python -m hymeko_rl.render_planar_gifs --checkpoint checkpoints/galambos/bc_ppo_mlp.pt    --kind mlp    --algo ppo --hand-authored --label mlp_bc    --run policies --difficulty 0.3 --max-steps 300 --seeds $SEEDS
cp reports/gifs/policies/*.gif docs/manipulation_models/results/gifs/ 2>/dev/null

# 3. FANUC BC->DDPG/TD3 WITH the warm-start fix (the real preserve/improve numbers)
for algo in ddpg td3; do
  for kind in hsikan mlp; do
    run fanuc-$algo-$kind 2400 python -m hymeko_rl.pick_place_bc --kind $kind --algo $algo --refine 12000 --n-demos 24 --n-epochs 80 --save checkpoints/fanuc/warmfix_${algo}_${kind}.pt
  done
done

echo "" >> "$Q"; echo "=== ALL DONE $(date +%H:%M:%S) ===" >> "$Q"
