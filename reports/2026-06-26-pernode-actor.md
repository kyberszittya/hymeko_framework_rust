# Per-node actor — the RL payoff test (the readout fix unlocks HSiKAN, but doesn't pass the MLP yet)

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-26-pernode-actor/` (tex/pdf/tikz/mmd) · **Status:** built, tested, first run done;
4-config highway run in flight.

## Summary

The supervised multidim probe (`2026-06-26-multidim-readout.md`) showed that for a per-joint output the SAC
actor's mean-pool → `Linear(feat, action_dim)` (collapse-then-expand) is the wrong shape, and a **per-node head**
(each joint's μ from its own message-passed node embedding + global context) is right. This is the RL test of
that on real control. Implemented `PerNodeActionHead` / `PerNodeSquashedGaussianActor` (actor-only; critic keeps
the global pool), `actuator_vertices` (joint→vertex map), and a parallel 3-config A/B.

## Result — galambos, SAC, 20k steps, **single seed**, params ~matched

| config | delivery | mean return | deaths | params |
|---|---|---|---|---|
| **mlp** | **0.45** | −116 | 4 | 15296 |
| hsikan_pooled | **0.00** | −177 | 0 | 15368 |
| hsikan_pernode | **0.20** | −130 | 0 | 15106 |

Artifacts: `experiments/2026_06_26_20_54_pernode_galambos/` (scoreboard.png, 3 GIFs, JSON, CSV).

### Reading (measured vs inferred)

- **Measured — the per-node readout is load-bearing: 0% → 20%.** The mean-pool actor delivers *nothing* (it is
  effectively inert — return −177 is just the time/distance penalties, 0 deaths because it barely moves the
  coin); the per-node actor controls the arms and delivers **20% cleanly (0 deaths)**. This is the RL
  confirmation of the probe's prediction: the collapse readout was discarding the actor's per-joint signal.
- **Measured — per-node HSiKAN does NOT beat the MLP yet** (0.20 vs 0.45). The MLP is more aggressive (delivers
  more, but **4 deaths** — it knocks the coin out); the HSiKAN policies are conservative (0 deaths, fewer
  deliveries). Different behavioural profiles, MLP ahead on raw delivery.
- **Inferred — the architecture story is "readout unlocks HSiKAN," not "HSiKAN wins."** The fix moves the actor
  from useless to functional; closing to/beating the MLP is unproven.

### Caveats (no overclaiming)

1. **Single seed** — per the 2026-06-26 determinism policy (CLAUDE.md §3) a single RL run is not a verdict; the
   0.20-vs-0.45 gap and the 0.00-vs-0.20 jump both need **multi-seed median/IQR**.
2. **Highway α-gate was OFF** (`skip="none"`) — the HSiKAN feature-collection gate (the "H") was not engaged.
   The 4-config run now in flight (`experiments/...pernode_galambos`, task `bl6yy7zfx`) adds
   `hsikan_pernode_hw` (per-node + highway, ~20.4k params) to test whether the gate closes the gap.
3. **20k steps / conservative exploration** — both may leave HSiKAN delivery on the table (the HSiKAN policies'
   0 deaths suggest caution; more exploration or steps could raise delivery). Exploration caveat from
   `project-hsikan-loses-possible-bug` applies, though at 20k the run is NOT a tie-at-failure (MLP 45%).

## Files (CORE.YAML: none — actor-only, non-core)
- `hymeko_rl/policy.py` (`PerNodeActionHead`), `hymeko_rl/sac.py` (`PerNodeSquashedGaussianActor`,
  `_SquashedGaussianActorBase`, `build_sac actor_head`/`skip`), `hymeko_rl/env/arm_world.py`
  (`actuator_vertices`), `hymeko_rl/exp_pernode_actor_ab.py` (parallel A/B + skip axis),
  `hymeko_rl/evaluate.py` (`experiment_dir`/`results_to_csv`; `plot_scoreboard` 3+-source fix).
- Tests: `test_pernode_actor.py` (6), `test_evaluate_stamp.py` (+3). All pass; ruff + mypy clean.

## Next
1. **4-config highway run** (in flight) — does `hsikan_pernode_hw` beat per-node / the MLP?
2. **Multi-seed** the winning config(s) for a real verdict (median/IQR), parallelised per the new policy.
3. If still short of the MLP: exploration (target_entropy/start_steps) and/or structural reward (task_graph/HTL).

## Provenance
Reproduce: `python -m hymeko_rl.exp_pernode_actor_ab --task galambos --mode full`. Git `fix-hsikan`, tree dirty.
Windows 11, Python 3.12, torch CPU. Single seed 0, 20k steps, 3 configs parallel (1 BLAS thread/worker).
