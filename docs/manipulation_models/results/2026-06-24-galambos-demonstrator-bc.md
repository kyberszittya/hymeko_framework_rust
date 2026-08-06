# Galambos demonstrator + behaviour cloning — overnight push

2026-06-24 · hymeko_rl · follows reports/2026-06-24-galambos-hyperedge-ab.md (the exploration-wall finding)

## Goal
Get a Galambos policy that actually delivers the coin, by breaking the hard-exploration wall with a scripted
demonstrator → behaviour cloning → PPO refinement (pure PPO never delivers, even with the coin at the zone).

## What shipped (all non-core, tested)
- **`planar_2link_ik`** — analytic IK for one planar 2-link arm; validated against the model (fingertip within
  1 cm on both arms; 3 tests).
- **`GalambosDemonstrator`** — scripted corral→pinch→carry on both arms (each arm on its own x-side, the natural
  geometry), built on the IK, reading arm geometry from the model (works for hand-authored and emitted robots).
- **`galambos_bc.py`** — `collect_galambos_demos` (roll the demonstrator, keep successful trajectories) +
  `behaviour_clone` (reused) + optional PPO refine + `eval_delivery`. CLI.

## Findings
**1. The Galambos task is hard at the CONTROL level, not just exploration.** The demonstrator reliably grips the
coin (~10/12 reach the carry phase) but only **delivers ~25%**. Cause: the coin is a free-spinning cylinder
(`disk_rz` hinge); a 2-finger clamp dragged *perpendicular* to its axis lets the coin **roll out**, and the two
arms (rooted at ±0.14 in x) cannot both get behind a central coin to *push* it. Tried and measured: gap sizing
(coin-radius-aware), hard clamp (high normal force), slow carry (`pull_step` 0.012→0.003), tight pre-pinch — all
top out ~2–3/12. Deliveries happen mainly when the coin is x-offset from the zone (the clamp drags well in x,
not in y). This is the same difficulty that defeated pure RL — *both* RL and a careful scripted controller cap
around 25%.

**2. BC reproduces the demonstrator.** Cloning the successful trajectories (HSiKAN, 120 demos → ~4.2 k
transitions, BC loss 7.8e-4) gives **20.8 % greedy delivery** — i.e. BC successfully transfers the approach→grip→
carry structure to the policy (the exploration wall is broken; the policy now *attempts* the full task).

## In flight (overnight)
BC→PPO comparison, **HSiKAN vs MLP × 3 seeds** at difficulty 0.3:
`for kind in hsikan mlp; for seed in 0..2: galambos_bc --kind $kind --seed $seed --n-demos 300 --bc-epochs 250
--ppo-iters 120`. Log: `reports/2026-06-24-galambos-bc-ppo.log` (one JSON per cell, partial-survivable). ~3 h.
**Question:** does PPO refine the carry beyond the demonstrator's 25 % (RL may find a wiggle/push the scripted
form can't express), and does HSiKAN beat a params-matched MLP once both start from a learnable BC init? Both
outcomes are informative.

## Honest framing for Kato
- The **FANUC pick-and-place is the working demo** (scripted expert 10/10 grasp, 9–10/10 place; GIF delivered) —
  lead with that.
- **Galambos is a genuinely hard manipulation problem** at the control level (rolling coin, two cooperating
  arms). The honest story: HyMeKo describes the robot+task; pure RL can't crack it; a demonstrator + BC breaks
  the exploration wall; the remaining gap is the physical carry — an open, interesting problem, not a polish item.
- This is also the strongest evidence yet for the **FSM / reward-machine line**: structure (declared phases +
  demonstrations) is what makes long-horizon manipulation learnable; flat RL on a scalar reward does not.

## Results (completed 2026-06-24, overnight run + auto-orchestrated queue)
**BC→PPO, HSiKAN vs MLP × 3 seeds @ diff 0.3** (greedy delivery rate):
- HSiKAN: BC 0.04–0.21 → **PPO 0.21 / 0.25 / 0.29** (median 0.25).
- MLP:    BC 0.08      → **PPO 0.17 / 0.21 / 0.25** (median 0.21).
HSiKAN edges MLP (0.25 vs 0.21 median) but **within the seed spread = effectively a tie**; both pinned at the
~25 % control ceiling. PPO refines BC modestly and holds there.

**BC→TD3 (off-policy / DDPG-family)** — *unreliable*: HSiKAN 0.083→0.125 (slight help), **MLP 0.292→0.083 (large
degradation)**. Off-policy from a BC warm-start collapses here (cold critic pulls the actor off the BC solution).
So TD3/DDPG is **not** the fix; PPO refine was at least stable. Checkpoints: `checkpoints/galambos/bc_td3_*.pt`.

**Conclusion (well-supported now):** Galambos caps ~25 % delivery **regardless of architecture (HSiKAN≈MLP) or
algorithm (PPO stable, TD3 erratic)**. BC breaks the exploration wall; nothing breaks the **physical carry
ceiling** (rolling coin). Strongest evidence yet that the fix is the **FSM / reward-machine** structure, not more
representation or RL tuning.

**Artifacts:** 4 timestamped demonstrator-success GIFs (`reports/gifs/demonstrator/demo_seed_{0,9,10,14}_goal.gif`);
the model-composition hypergraph (`reports/figures/composition.dot` — render with graphviz, not installed here).

## Follow-up
- On completion: parse `reports/2026-06-24-galambos-bc-ppo.log` for the BC-vs-PPO and HSiKAN-vs-MLP deltas.
- Add a demonstrator delivery-rate guard test (deferred — needs CPU free; the run is using it).
- If PPO refines well, the coin/zone-in-graph A/B (`task_graph=True`) finally becomes readable on a delivering
  policy. If not, the FSM line (`project-fsm-structured-rl`) is the principled next step.
