---
title: Aibo quadruped — restored, standing re-verified, walking extension
date: 2026-07-16
status: restored + re-verified (standing) + scripted walking demonstrator; RL walk is next
core_yaml_touched: none
supersedes_context: reports/2026-07-07-session-handoff-aibo-standing-dagger.md
---

# Aibo quadruped — restore, re-verify, walk

## Summary

Brought the **22-DOF Sony Aibo ERS-1000** scenario back (it had been replaced/deleted by uncommitted
working-tree changes on `integration/fanuc-pick-place-canonical`), re-verified the standing scenario, and
extended it toward **walking** with a scripted gait demonstrator + a confirmed RL path.

## Restore (branch hygiene)

The current working tree had `quadruped.hymeko` replaced with a simplified **4-actuator** quadruped and the
stand-scenario + strategy-graph files **deleted** (uncommitted, not from this session). The faithful Aibo is
committed at HEAD, so:

1. Backed up the working-tree 4-DOF `quadruped.hymeko` → `<scratchpad>/quadruped_working_4dof.hymeko`.
2. `git checkout HEAD -- data/robotics/{quadruped, quadruped_stand, meta_strategy_graph,
   quadruped_stand_bc_graph, quadruped_stand_dagger, quadruped_stand_td3bc_graph}.hymeko`.

Result: `quadruped.hymeko` back to 140 lines (22-DOF ERS-1000), env builds with **12 leg actuators**
(obs 33×2, nbody 34). No CORE.YAML touched. *The 4-DOF simplification is preserved in the backup;
restoring it discards other uncommitted work on this branch — flagged to and approved by the user.*

## Re-verification (standing) — on-record result intact

- **69 committed Aibo/strategy/scenario tests pass** (`test_quadruped_aibo_plant`, `test_quadruped_from_hymeko`,
  `test_quadruped_standing`, `test_strategy_graph`, `test_scenario*`, …) — the restore is clean.
- `QuadrupedGoalEnv.from_hymeko()` builds the standing MDP (task=stand, 12 actuators, max_steps 250).
- The scripted **PD-hold demonstrator holds 250/250 standing steps, upright 1.00** — the 1.0 imitation ceiling
  from the 2026-07-07 handoff reproduces.
- The measured standing result is **on record** (kato15 RTX 6000 Ada, 3 seeds): PD-hold 1.0 → BC 0.458 →
  **TD3+BC 0.0 (off-policy value drift, collapses)** → **DAgger 0.958**. Not re-run here (GPU-scale, on record);
  the local checks confirm the pipeline is intact. Verdict stands: *imitation (DAgger) is the lever past a BC
  ceiling, not off-policy RL.*

## Walking extension (new)

- **`QuadrupedTrotGait`** (`hymeko_rl/env/locomotion_experts.py`): a closed-loop **PD trot** — PD-tracks a
  diagonal-trot joint trajectory (offsets the standing pose `q0`; {fl,br} anti-phase to {fr,bl}; knee
  anti-phase to hip). It walks the Aibo **forward, upright, finite** (dx +0.38 m over 700 steps toward the
  goal). Rendered: `reports/figures/2026-07-16-aibo/aibo_walk.mp4` (the dog trotting on the checker floor).
- **Honest limit:** it's a **slow shuffle (~0.05 m/s)**. A naive open-loop torque CPG drifted *backward*;
  PD tracking gives balance but hand-tuned gaits don't produce fast clean walking — exactly the standing
  lesson (scripts hit a low ceiling; RL/DAgger is the lever). The gait is a *demonstrator / BC anchor*, not
  the deliverable.
- **RL path confirmed:** TD3 (`train_offpolicy`) trains on the Aibo goal-reach task out of the box (tensor-
  contract passes, live progress logs). The goal task is jump/lunge-capable under RL — the route to a real
  fast walk, mirroring how standing needed DAgger.

## Files touched

New (non-core): `hymeko_rl/env/locomotion_experts.py` (+`QuadrupedTrotGait`), `hymeko_rl/tests/test_aibo_walk.py`
(4 tests), `reports/figures/2026-07-16-aibo/aibo_walk.mp4`, this report.
Restored (to committed state): the 6 `data/robotics/quadruped*/meta_strategy_graph.hymeko` files.

## Test results

- **41 tests pass** (`test_aibo_walk` 4 + `test_locomotion_env` 37); **69** committed Aibo/strategy tests pass.
  ruff clean.
- Coverage: `QuadrupedTrotGait` — forward-walk, upright, shape/bounds, determinism.

## CIP / coin-toss / Peter enhancement — cheap loop (mechanism proven, honest smoke)

Wired three research findings into one Aibo goal-reach loop (`experiments/exp_aibo_cip_walk.py`,
`experiments/2026_07_16_aibo_cip_walk/smoke.json`): **CIP** (a DirectLiNGAM diagnosis of the trot →
new `vertical_bounce` reward term `−v_z²`; then **re-diagnose** the learned policy), **coin-toss** (trot as
BC anchor; asymmetric CTDE), **Peter** (structural/relational actor over the leg-hypergraph; privileged critic).

CIP found the trot's dominant edge is **leg_speed ⇒ torso_height +0.50 (bounce)**, not **⇒ forward_vx +0.12
(propel)** — a *causal* explanation of the slow shuffle. Cheap smoke (4k-step TD3+BC, CPU):

| actor | propel edge (leg⇒vx) trot→learned | bounce edge trot→learned |
|---|---|---|
| flat mlp | 0.12 → **−0.10** (degraded) | 0.45 → 0.55 |
| **structural (hsikan)** | 0.12 → **+0.155** (improved) | 0.45 → 0.70 |

**Findings:** (1) the loop + CIP reward + asymmetric critic + structural actor all run end-to-end; (2) the
**structural (relational) actor raised the causal propel-edge where the flat MLP degraded it** — a small but
real validation of Peter's *relational-belief-is-load-bearing* finding, even in a collapse-prone smoke; (3)
neither reduced bounce or netted forward distance in 4k steps, and the log shows `act loss → nan` — i.e.
**TD3+BC value drift**, re-confirming the coin-toss / Aibo-standing lesson that off-policy is *not* the lever.
Mechanism proven; the *improvement* needs the real campaign (DAgger, structural actor, CIP-shaped reward,
multi-seed, kato15), where the CIP propel-vs-bounce edge is the success metric.

New: `env/reward.py` (+`vertical_bounce` term), `experiments/exp_aibo_cip_walk.py`, +1 test.

## Open / next

- **RL walking** — SAC/TD3/DAgger on the Aibo goal-reach task (the real fast walk); wire the trot as the BC
  anchor. Best on kato15 (the handoff's `.venv_stand` + `launch_*.sh`; GPU ~1135 steps/s).
- **Qt sim** — add the Aibo (walk + stand) to `hymeko_rl/gui/vehicle_qt.py` for live interactive viewing.
- **Terrain** — the dog on a heightfield (rough-terrain walking) once a stronger gait/policy exists.
- Untouched threads from the 2026-07-07 handoff still stand (Gazebo output PLANNED-not-built; DAgger clean
  curve — do not chase; best-checkpoint 0.958 is robust).

## Provenance

Branch `integration/fanuc-pick-place-canonical`; restored files at committed state, new files uncommitted.
Env: Python 3.11 `.venv`, mujoco 3.10.0, macOS Apple-Silicon CPU. Seeds: 0 (env determinism, TD3 smoke).
Standing/DAgger numbers cited from `reports/2026-07-07-{aibo-quadruped-hymeko-substrate, session-handoff-…}.md`
(kato15), not re-measured.
