# Report — declarative observation: `node_features` driven by the `.hymeko` obs profile

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ✅ complete — the env's observation is now assembled from a declared
`ObservationSpec` (read from a `.hymeko` profile); the hand-coded column layout is gone.
Full `hymeko_rl` suite green (48 tests).

## Summary
Third increment of the Kato "one `.hymeko` source → scene + hypergraph + obs" line. The
observation was the last **procedural** piece — `node_features()` hard-coded the column
offsets (`feat[v,0]=qpos`, `[v,1]=qvel`, `[:,2:5]=target`, `[:,5:8]=ee_error`). It is now a
**declarative channel spec**: an ordered list of channel kinds, each mapped to a width and a
live-state extractor (Strategy), assembled into the `(N_vertices, Σ dim)` tensor. The spec
is read directly from a `.hymeko` observation profile, so a new observation needs only a new
`.hymeko` — no Python change.

The behaviour is **byte-identical** to the former layout (an equivalence test asserts it), so
every existing policy/BC/PPO test passes unchanged.

## Design (data-oriented + Strategy)
New module **`hymeko_rl/env/observation.py`**:
- **Channel registry** `_CHANNELS: dict[str, _ChannelImpl]` — each kind bundles its `dim`
  and `extract(env)` (the dim and the Strategy together, so they cannot drift). Implemented
  kinds: `joint_position` (1), `joint_velocity` (1), `target_position` (3), `ee_error` (3);
  the per-joint scatter is shared by one helper (`_scatter_joint_scalar`), the globals by one
  (`_broadcast`) — no per-channel duplication.
- **`ObservationSpec(channels: tuple[str, ...])`** — frozen; `obs_dim` = Σ channel dims;
  `assemble(env)` = `concatenate([extract(env) for each channel])`. `__post_init__` rejects
  empty / unknown channels (DbC).
- **`read_channels(profile)`** — a narrow reader for the regular profile form
  (`@name: ns.kind { (+ a, + b …); }`): pairs each `observation_space` member with its
  declared kind, preserving the arc's order. A documented **bridge** until the engine's
  import-resolving snapshot reaches Python (B-003) — there is no structured IR dump from the
  CLI (`--json` exists only on `entropy`/`rewrite`) and the PyO3 `snapshot_json` is
  B-003-blocked, so this profile-shape parse is the robust option for now.
- **`REACH_OBSERVATION`** — the reaching default (the 4 channels above), used when no profile
  is supplied.

Wiring:
- **`arm_reach_env.py`** — `node_features()` is now `return self.obs_spec.assemble(self)`;
  `__init__` takes `obs_spec` (default `REACH_OBSERVATION`); `observation_space` is shaped by
  `obs_spec.obs_dim`; `from_hymeko` gains `obs_profile=` (→ `ObservationSpec.from_hymeko`),
  completing one-source scene **and** obs. `_NODE_FEAT` is retained but now *derived*
  (`= REACH_OBSERVATION.obs_dim`), so it can't drift from the spec.
- **`bc.py`** — sizes the policy from `env.obs_spec.obs_dim` (not the module constant), so a
  non-default obs profile is handled without touching BC.
- **`agent.py`** — docstring refreshed; `ObservationSpec.from_hymeko` is the channel reader
  `AgentSpec.from_hymeko` will compose with the kinematic vertex count.

## Files touched (untracked `hymeko_rl/` package)
| File | Change |
|---|---|
| `hymeko_rl/env/observation.py` | **new**, ~150 — extractors, registry, `ObservationSpec`, `read_channels` |
| `hymeko_rl/env/arm_reach_env.py` | spec-driven `node_features`, `obs_spec` param, obs shape, `from_hymeko(obs_profile=)`, derived `_NODE_FEAT` |
| `hymeko_rl/bc.py` | policy sized from `env.obs_spec.obs_dim`; dropped `_NODE_FEAT` import |
| `hymeko_rl/agent.py` | docstring (obs now wired) |
| `hymeko_rl/tests/test_observation.py` | **new**, ~95 — 7 tests |

**CORE.YAML items touched:** none.

## Tests (`pytest -p no:randomly`)
- **`test_observation.py` — 7 passed:** spec dims/validation; `read_channels` recovers the
  reaching profile's kinds and matches the default spec; reader rejects a missing
  `observation_space`; reader preserves the **arc order** (not decl order); and an
  **equivalence** test — the spec-driven `node_features` is `np.allclose` to an independently
  reconstructed legacy layout.
- **Regression — 41 passed, behaviour unchanged:** `test_reach_bc` + `test_arm_reach_from_hymeko`
  + `test_arm_world` (21) and `test_policy` + `test_ppo` + `test_hypergraph_state` (20).

End-to-end smoke (one source): `ArmReachEnv.from_hymeko(anthropomorphic_arm.hymeko,
obs_profile=arm_reach_observation.hymeko)` → channels read from the profile
`(joint_position, joint_velocity, target_position, ee_error)`, obs `(7, 8)`, 6 actions,
expert reaches `0.490 → 0.057 m` (12%).

## Static analysis
- `ruff check` (4 changed files) → **clean**.
- `mypy` → only `mujoco` `import-untyped` (transitive, pre-existing baseline — `hypergraph_state.py`
  has it identically); none in `observation.py`, none on my annotations.
- **§6.5 anti-patterns:** none. The channel families are unified by Strategy + a registry (not
  per-channel functions — #1/#9); the per-joint scatter and the broadcast are each one shared
  helper (#3); dim+extractor are bundled so they can't drift (no string-typed dim duplication).

## Performance
Not a perf change. `assemble` does the same scatter/broadcast/concatenate the old code did
(N is ~7); equivalence test confirms identical output. `read_channels` runs once at env
construction.

## Dependencies
None added or removed.

## Open issues / follow-ups
1. **`AgentSpec.from_hymeko`** — the MDP-level wrapper can now use `ObservationSpec.from_hymeko`
   for the per-vertex `obs_dim`; it still needs the kinematic vertex count (from the model) and
   the action/reward profile to produce a full `AgentSpec`.
2. **Reward declarative** — still procedural (`−dist`). Folding it into the `.hymeko` (over
   `meta_task`) is the remaining piece to make the *whole* MDP one source.
3. **Structured profile read** — `read_channels` is a narrow text parse; replace with the
   engine snapshot once B-003 (PyO3 import resolver) lands. The vertex *bindings* in the
   profile (which joints a channel covers) are not yet consumed — the env scatters over all
   model joints; honouring per-channel vertex sets is a future refinement.

## Provenance
- Git: `hymeko_rl/` untracked; this task added `observation.py`, `test_observation.py` and
  edited `arm_reach_env.py`, `bc.py`, `agent.py`.
- Platform: Windows 11, MuJoCo 3.9.0, Python 3.12.
- Fixtures: `data/robotics/anthropomorphic_arm.hymeko`, `data/robotics/arm_reach_observation.hymeko`.
- Seeds: equivalence test seed 3; reach smoke seed 2.
