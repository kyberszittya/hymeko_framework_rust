#!/usr/bin/env bash
# Orchestrator: wait for the overnight Galambos BC+PPO run to finish, then run the morning queue SEQUENTIALLY
# (no torch concurrency). Each step is best-effort: a failure is logged and the queue continues.
cd "d:/hakiko_ai_ws/03_implementation/hymeko_framework_rust" || exit 1
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=.
RUNLOG=reports/2026-06-24-galambos-bc-ppo.log
Q=reports/2026-06-24-morning-queue.log

echo "[orchestrator $(date +%H:%M:%S)] waiting for overnight run (ALL DONE in $RUNLOG)..." > "$Q"
for _ in $(seq 1 420); do
  if grep -q "ALL DONE" "$RUNLOG" 2>/dev/null; then
    echo "[orchestrator $(date +%H:%M:%S)] overnight run finished — starting queue" >> "$Q"; break
  fi
  sleep 60
done

step() { echo "" >> "$Q"; echo "=== $1 $(date +%H:%M:%S) ===" >> "$Q"; shift; "$@" >> "$Q" 2>&1 && echo "[ok]" >> "$Q" || echo "[FAILED]" >> "$Q"; }

# 1. overnight BC+PPO results summary
{ echo ""; echo "=== overnight BC+PPO results ==="; grep -E '=== |delivery' "$RUNLOG"; } >> "$Q" 2>&1

mkdir -p reports/figures
# 2. model composition as a hypergraph (DOT, + SVG if graphviz present)
step "composition-hypergraph" python -m hymeko_rl.hymeko_compose data/robotics/pick_place_scenario.hymeko --dot reports/figures/composition.dot
command -v dot >/dev/null 2>&1 && dot -Tsvg reports/figures/composition.dot -o reports/figures/composition.svg >> "$Q" 2>&1

# 3. simulation views: the scripted demonstrator's SUCCESSFUL deliveries (timestamps auto-baked)
step "demonstrator-gifs" python -m hymeko_rl.render_planar_gifs --demonstrator --difficulty 0.3 --out-root reports/gifs --run demonstrator

# 4. DDPG: BC -> off-policy (TD3) refine, HSiKAN vs MLP on Galambos (does off-policy beat PPO's ~25% ceiling?)
step "bc-td3-hsikan" python -m hymeko_rl.galambos_bc --kind hsikan --algo td3 --refine 20000 --n-demos 300 --bc-epochs 250 --difficulty 0.3 --save checkpoints/galambos/bc_td3_hsikan.pt
step "bc-td3-mlp"    python -m hymeko_rl.galambos_bc --kind mlp    --algo td3 --refine 20000 --n-demos 300 --bc-epochs 250 --difficulty 0.3 --save checkpoints/galambos/bc_td3_mlp.pt

echo "" >> "$Q"; echo "=== QUEUE DONE $(date +%H:%M:%S) ===" >> "$Q"
echo "(deferred — need attended wiring: FANUC BC->DDPG, TD3-policy GIFs [DeterministicActor loader], HSiKAN-arch-as-hypergraph)" >> "$Q"
