# Proposal — the observation / state space as a HyMeKo hypergraph (declarative obs tensor)

**Date:** 2026-06-18
**Status:** ✅ vocabulary + example authored and **validated** (`hymeko validate` / engine
parse); the declarative observation compiles to the IR whose star-expansion is the obs
tensor. `AgentSpec.from_hymeko` (the in-memory bridge) is the one wiring step left.

## The idea (answering "where do we propose the tensor obs / state space")
Today the policy's observation is **procedural** — hand-coded in
[hymeko_rl/env/arm_reach_env.py](../hymeko_rl/env/arm_reach_env.py) `node_features()`:
per kinematic vertex `[qpos, qvel, target_xyz, ee_error_xyz]` = 8 features. This proposal
makes the obs/state space a **declarative HyMeKo artifact** — a fourth `.hymeko`
vocabulary alongside `meta_kinematics` (geometry) and `meta_task` (behaviour), so **one IR
carries the structure AND the agent's observation**, and the obs *tensor* is just the
star-expansion of that observation hypergraph (the bridge `hymeko_rl.HypergraphState`
already does). This is exactly Csaba's framing to Kato: *"the observation and state space
as a hypergraph with an equivalent tensor representation."*

## What was built (and validated)
**`data/robotics/meta_observation.hymeko`** — the vocabulary (parses via the engine, 12
types). It declares:
- **per-vertex feature channels** (`@joint_position` dim 1, `@joint_velocity` 1,
  `@joint_effort` 1, `@link_pose` 3, `@link_twist` 6) — each a hyperedge `(+ vertex_set)`;
- **global channels** (`@target_position` 3, `@ee_error` 3, `@command` 1) — broadcast;
- **`@observation_space`** — a bundle of channels; its star-expansion is the
  `(N_vertices, Σ dim)` obs tensor.

**`data/robotics/arm_reach_observation.hymeko`** — the example (validates via the CLI). The
declarative form of `node_features()`:
```
@qpos: feat.joint_position { (+ j1, + j2, + j3, + j4); }
@qvel: feat.joint_velocity { (+ j1, + j2, + j3, + j4); }
@target: glob.target_position { (+ target_frame); }
@err:    glob.ee_error        { (+ flange, - target_frame); }
@reach_state: obs.observation_space { (+ qpos, + qvel, + target, + err); }
```
`hymeko inspect` confirms `reach_state` compiles to a hyperedge over
`(+qpos,+qvel,+target,+err)`, and `err` to `(+flange, -target_frame)`. The channel dims sum
to **1+1+3+3 = 8** — identical to the current `_NODE_FEAT = 8`. So the procedural obs and
the declarative obs are the *same* tensor; the `.hymeko` is the source of truth.

## The bridge (the one wiring step left)
`hymeko_rl/agent.py::AgentSpec.from_hymeko(path)` should: parse the obs profile → read
`observation_space`'s member channels → sum their `dim`s → `obs_dim`; the *structure*
(which vertices) is the kinematic hypergraph `HypergraphState` already extracts. Then the
env builds `node_features()` from the declared channels instead of hard-coding them — and a
new robot/observation needs **no Python change**, just a new `.hymeko`.

**Dependency:** the robust path reads the engine's `snapshot_json` of the *imported* obs
profile, which is currently blocked by **B-003** (the PyO3 import resolver — see
`docs/BUGS.md`); the CLI resolves imports (`validate`/`inspect` both work), so `from_hymeko`
can route through the CLI meanwhile, or land cleanly once B-003 is fixed.

## Why this matters (Kato)
- **One source, three products:** the same robot `.hymeko` (+ profiles) emits the **MJCF
  scene**, the **kinematic hypergraph**, and now the **observation/state space** — all as
  hypergraphs compiled to tensors via star-expansion. No procedural special-case.
- **Composes with the existing lines:** `meta_observation` is the fourth role of the
  unification (the *agent description*); it reuses the same IR, star-expansion, and
  `HypergraphState` bridge as the geometry and the policy.
- **Scales:** the quadruped reuses it unchanged — its richer obs (per-leg joint state +
  contact channels) is just more channels over more vertices.

## Files touched
New: `data/robotics/meta_observation.hymeko`, `data/robotics/arm_reach_observation.hymeko`.
(Both validate; no code change yet — `AgentSpec.from_hymeko` is the next increment.)
**CORE.YAML:** none. **No new dependency.**

## Next
1. Wire `AgentSpec.from_hymeko` (via the CLI now, or the engine snapshot once B-003 lands).
2. Drive `node_features()` from the declared channels (procedural → declarative).
3. Extend the profile to **action + reward** (the full MDP as `.hymeko`), and author a
   quadruped observation profile (contact channels) for the locomotion line.
