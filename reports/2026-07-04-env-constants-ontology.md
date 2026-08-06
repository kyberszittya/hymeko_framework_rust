# Env constants → ontological hierarchy (Phase 1)

**Date:** 2026-07-04 · **Branch:** `hymeko-neuro-migration` · **Plan:** `docs/plans/2026-07-04-env-constants-ontology/` (4 artifacts)

## Summary

Extracted the magic constants scattered across the `hymeko_rl/env` MJCF builders into a single
`hymeko_rl/env/constants.py`, organised **ontologically** — grouped by *what kind of quantity* each
is (a taxonomy in namespaced classes), not by which file used it:

- **`Physics`** — `TIMESTEP`, `STABLE_DT`, `GRAVITY`, `INTEGRATOR` (+ `gravity_attr()`,
  `option_attrs()` string helpers). Was duplicated verbatim across 4 builders.
- **`Collision`** — the `contype`/`conaffinity` bitmask scheme as a checked `IntEnum`
  (`Type`, `Affinity`) plus named `(contype, conaffinity)` **channels** (`FINGERTIP`, `COIN`,
  `FLOOR`, `ARM_DEFAULT`, `VISUAL`) and the MuJoCo `collide()` predicate. The meaning now lives in
  the type, not a comment.
- `Geometry` (per-world sizes) deferred to Phase 2; `Reward` left as-is (already named `RewardSpec`
  tuples).

Behaviour-preserving: every emitted number is identical — the string helpers reproduce the old
literals exactly (`Physics.option_attrs()` → `timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"`,
`Collision.attr(FINGERTIP)` → `contype="1" conaffinity="3"`).

## Files touched

- **New:** `hymeko_rl/env/constants.py` (+94), `hymeko_rl/tests/test_env_constants.py` (+66).
- **Modified (literal → named ref):** `planar_grasp_env.py` (6 sites), `arm_world.py` (1),
  `gripper_world.py` (2), `pick_place_env.py` (1). Import added to each.

## CORE.YAML

None touched.

## Test results

| Layer | Test | Result |
|-------|------|--------|
| Unit — values-parity | constants equal the replaced literals | pass |
| Unit — collision-scheme | `collide()` over channels: coin↔fingertip/floor True, coin↔arm **False**, finger↔finger True; mask-typo regression | pass |
| Integration — golden guard | emitted MJCF carries byte-identical option/collision fragments and compiles to a valid `MjModel` | pass |
| Regression | `test_env_constants` + `test_planar_grasp_env` = **34 passed**; pick/gripper/arm/quadruped = **66 passed** | pass |

Static: `ruff check` on all 6 changed files — **All checks passed**. Complexity: every function
< 15 LOC; no §6.5 anti-patterns (the `IntEnum` replaces a string-typed/magic-int scheme per #7).

## Collision semantics (verified, not assumed)

MuJoCo collides `A,B` iff `(contype_A & conaffinity_B) | (contype_B & conaffinity_A) != 0`. The coin
is on **bit 2 only** (`2/2`); arm links use the MuJoCo default `1/1` → `coin & arm = 0`, so arm
bodies cannot touch the coin (only fingertips `1/3` and floor move it). This is now asserted by
`test_collision_scheme_isolates_coin_from_arm_links`, so a future mask edit that breaks it fails CI.

## Open / follow-ups

- **Phase 2 (Geometry):** per-world geom sizes/radii (fingertip 0.014, coin 0.022×0.012, floor
  extents, pedestal, gripper finger dims) — inherently local, lower value; deferred.
- Not yet committed alongside the merge — will commit on the `hymeko-neuro-migration` branch.

## Provenance

Host win32, Python 3.12.13, pytest-8.4.2, mujoco (CPU MJCF compile). No stochastic runs (pure
refactor). No seeds required.
