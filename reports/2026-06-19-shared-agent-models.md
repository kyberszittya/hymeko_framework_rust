# Report — shared reward model + import-aware reader + AgentSpec.from_hymeko (phase 2)

**Date:** 2026-06-19 · **Plan:** `docs/plans/2026-06-19-shared-agent-models/` + the xprofile core enabler
**Status:** ✅ **Done.** The reach reward is authored **once** and reused across both task descriptions
via a cross-profile reference; the Python reader follows imports to match the compiler; and
`AgentSpec.from_hymeko` composes the MDP from the profiles. Builds on the core enabler
(`APPROVED-CORE-EDIT: xprofile-instance-refs`).

## Summary
Consumes the cross-profile instance-reference capability. The reach reward (`flange`, `target_frame`,
`@goal`, `@dist`) — previously copy-pasted into `arm_reach_task` and `arm_reach_safe_task` — now lives
**only** in a shared model, referenced cross-profile as `arr.dist`.

## Changes
- **`data/robotics/arm_reach_reward.hymeko`** (NEW) — the shared reach reward model (frame bindings +
  `@goal` + `@dist`, weight 1.0).
- **`arm_reach_task.hymeko`** / **`arm_reach_safe_task.hymeko`** — rewired: import the shared model,
  `using arm_reach_reward_description.arm_reach_reward as arr`, `reward_spec = (+ arr.dist[, +penalties])`.
  No more re-declared reach reward. Both also import the meta vocab the shared profile uses (phase-1
  cross-profile constraint: no transitive indexing yet — the importer supplies the namespaces).
- **`hymeko_rl/env/_profile.py`** — `read_bundle` is now **import-aware**: `_gather_decls` follows
  `@"…"` imports (cycle-guarded; local decls shadow imported), and a bundle member's last dotted
  segment is the decl name (`arr.dist` → `dist`). Mirrors the compiler — now *correct* to do, since
  `hymeko inspect` resolves the same refs.
- **`hymeko_rl/agent.py`** — `AgentSpec.from_hymeko(robot, obs_profile, task_profile, …)` composes
  `obs_dim` (per-vertex, from `ObservationSpec`) + `n_vertices` (kinematic hypergraph) + `action_dim`
  / bounds (emitted model actuators) + `reward` (task-profile name). Added the `n_vertices` field.
- **Observation** needed no change — `arm_reach_observation.hymeko` was already a shared standalone
  model; any agent reuses it by passing it as `obs_profile` (and `AgentSpec.from_hymeko` does).

## Verification
- **`hymeko inspect`** validates both rewired tasks; `dist` resolves cross-profile (same decl hash as
  before the refactor → semantics preserved).
- **Python specs identical (regression):** `RewardSpec.from_hymeko` → plain `(reach_distance,1.0)`;
  safe the same 5 terms; `TerminationSpec` `(ground_contact, self_collision)`. The safe-task read now
  exercises cross-profile resolution (`arr.dist` from the imported shared file) — which the old
  single-file reader could not do.
- **`AgentSpec.from_hymeko` parity:** equals an `ArmReachEnv` built from the same sources on
  `obs_dim`, `n_vertices`, `action_dim`, and action bounds (test `test_from_hymeko_matches_the_env`).
- **Suite:** `pytest hymeko_rl/tests/` **97 passed**; `ruff` clean; `mypy --strict` only the `mujoco`
  baseline. (Core enabler separately: `hymeko_core` 133, `hymeko_query` 212, hash-parity intact.)

## CORE.YAML / dependencies
This phase is **non-core** (`hymeko_rl/`, `data/robotics/`). The enabling core edit is reported and
token-bearing in `reports/2026-06-19-xprofile-instance-refs.md`. No dependency change.

## §6.5 anti-patterns
Resolves #3 (per-experiment scaffold duplication) for the reach reward — it's now a single shared
model. Import-following reuses one `_gather_decls` helper (no duplication); `AgentSpec.from_hymeko`
lazy-imports to avoid a cycle.

## Open / follow-up
- **Phase-1 constraint:** an importer must also import the meta vocab the shared profile uses (no
  transitive indexing). Documented in the rewired profiles. Lifting it (transitive import indexing)
  is a future core enhancement.
- **Mixin / bare-name reuse** ("when performance requires it") — a later, separately-approved core phase.
- **Cleaner ref path** (`shared.dist` without the `_description` segment) — needs path-flattening in
  the resolver; deferred.

## Provenance
- Git SHA `7d16ad0` (working tree dirty; uncommitted increment). MuJoCo 3.9.0 / torch per CORE pins.
  Windows 11. Deterministic (no seed in these tests).
