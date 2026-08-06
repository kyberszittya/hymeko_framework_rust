# R11.7A — HyMeKo-Generated Manipulation Architecture Unification (U1–U5)

**Date:** 2026-08-06 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher` (worktree `hymeko_coin_r9_wt`)
· **Base SHA (clean at start):** `8da74b8a`
**Status:** U1–U5 complete. **O0 parity gate PASSED bit-exactly** (re-confirmed after full `ObjectSpec` threading).
U6 (object variant campaign) pending — halted for scoping per user decision.

---

## Summary

R11.7A was reframed (user directive) from "thread a `coin_shape` param" into an **architectural unification**: the
executable manipulation environment and its structural contracts are **generated from one HyMeKo scene/object
specification**, eliminating the divergent reconstruction paths. This report covers the first four stages, through the
**blocking O0-parity gate**. The reference coin now flows from `galambos_env.hymeko` → a typed `ObjectSpec` → one
`build_manipulation_rig` generator → the frozen R11.6C stack, **bit-for-bit identical** to the pre-unification path.

- **U1** — architecture map + duplicate-ownership audit (KEEP/MOVE/ADAPT/DEPRECATE/DELETE):
  `reports/2026-08-06-r11-7a-u1-architecture-map.md`. Authoritative builder = `compose_planar_scene`; frozen = dynamics/
  policy, **not** geometry.
- **U2** — the object is a typed `ObjectSpec` read from the HyMeKo scene (`shape`/dims/mass/friction/damping/family),
  not split across Python kwargs.
- **U3a** — one canonical `build_manipulation_rig(object_spec, robot)` generator + a stable-handle `RigRegistry`;
  `_reconstruct` routed through it, `_make_env`'s hardcoded `coin_shape="cylinder", hx=0.020` literal **unpinned** to
  read the scene object.
- **U4** — O0 parity gate: generated path == frozen R11.6C, proven three ways (below).
- **U3b** — closed the density/friction plumbing gap by threading `ObjectSpec` as a **single carrier** through
  the whole chain (`reconstruct_handoff → replay_pi0 → CoinRL4Dof → neutral_env → make_coin_env`), not more loose
  kwargs; parameterized `CoinStraddleTargets.for_object(footprint)` (reproduces the coin's 0.055 standoff).
  **O0 parity re-confirmed bit-exact after the threading.**
- **U5** — duplicate-ownership guard (AST allowlist: `compose_planar_scene` is centralized; the 2 pre-R11.7A
  direct builders are pinned and may not grow; the bv `_make_env` literal stays unpinned) + deprecation notes on
  the loose object kwargs. Physical deletion deferred to post-U6 (user decision).

## Files touched

**New:** `hymeko_rl/env/object_spec.py` (typed ObjectSpec + Shape) · `hymeko_rl/coin_delivery/manipulation_rig.py`
(build_manipulation_rig + RigRegistry + RobotContract) · `hymeko_rl/experiments/r11_7a_o0_parity.py` (production smoke)
· tests: `test_object_spec.py`, `test_manipulation_rig.py`, `test_o0_parity_regression.py`, `test_object_ownership_guard.py`.
**Modified:** `data/robotics/galambos_env.hymeko` (full coin identity on `@dsk`) · `hymeko_rl/env/env_spec.py` (EnvSpec
HAS-A ObjectSpec) · `hymeko_rl/experiments/video_coin_variants.py` (`_reconstruct` routes through the generator; loose
kwargs deprecated) · `hymeko_rl/experiments/bv_identification_benchmark.py` (`_make_env` scene-sourced) · **U3b chain
(single-carrier `object_spec`):** `hymeko_rl/coin_delivery/coin_late_start.py` (`reconstruct_handoff`/`replay_pi0`),
`hymeko_rl/coin_delivery/coin_rl_env.py` (`CoinRL4Dof`), `hymeko_rl/experiments/coin_neutral_start.py` (`neutral_env`),
`hymeko_rl/coin_delivery/env_factory.py` (`make_coin_env`) · `hymeko_rl/coin_delivery/theta_option/planar_geometric_approach.py`
(`CoinStraddleTargets.for_object`) · `hymeko_rl/tests/test_planar_geometric_approach.py` (straddle test).
**CORE.YAML items touched:** none (all edits are Python + a `.hymeko` scene, none protected).

## Design (unification seam)

`galambos_env.hymeko` (`@dsk: p.disk { shape "cylinder"; family "coin"; radius 0.02; half 0.02; damping 2.5;
spin_damping 0.8; frictionloss 0.0; }`) → `EnvSpec.from_hymeko` (parses the object term via the string-aware
`parse_fields`) → `EnvSpec.object: ObjectSpec` → `build_manipulation_rig(object_spec, robot)` threads the object as a
**single carrier** through the reconstruct chain to `PlanarGraspEnv` (shape/size AND density/friction/damping — no axis
dropped), with the robot supplying `geom`/`arm_mjcf_transform`. `Shape` is a typed enum with a boundary parser
(string-in-HyMeKo → enum-in-Python). The manipuland geom/body handle is always `"disk"` (stable across shapes), verified
by `RigRegistry.resolve`. The straddle standoff is footprint-aware (`CoinStraddleTargets.for_object`), reproducing the
coin's 0.055 exactly.

## Test results

| Layer | Tests | Result |
|---|---|---|
| U2 unit (`test_object_spec.py`, incl. box+density endpoint) | 20 | pass |
| U2 regression (`test_galambos_scene_authority_sentinels.py`) | 10 | pass |
| U3 unit (`test_manipulation_rig.py`) | 8 | pass |
| U3b straddle (`test_planar_geometric_approach.py::…for_object`) | 1 | pass |
| U3b geometric-approach suite (straddle consumers) | 21 | pass (24 s) |
| U4 O0 parity smoke (`r11_7a_o0_parity.py`, 3 demos) | — | `O0_PARITY_PASS`, `max_dev=0` |
| U4 O0 parity regression (`test_o0_parity_regression.py`, `slow`) | 1 | pass (21.8 s) |
| U5 duplicate-ownership guard (`test_object_ownership_guard.py`) | 3 | pass |
| Broader chain regression (snapshot/capture/zero-home/torque-path) | 27 | pass (279 s) |
| Broader planar regression | 73/74 | 1 **pre-existing** failure (below) |

`ruff check` clean on all 9 changed non-test modules; `mypy --strict` clean on the clean-authored modules
(`object_spec.py`, `manipulation_rig.py`); the one mujoco `import-untyped` is a third-party-stub gap, scoped-suppressed
with a reason comment (§6.3). The 11 pre-existing E702 (`;`) warnings in `coin_late_start.py` are on unchanged lines
(present identically on the clean tree) — not introduced here.

**Pre-existing failure (not introduced here):** `test_planar_grasp_env.py::test_env_shapes_and_coin_placed_in_reach`
asserts that over 16 fixed seeds the coin *sometimes* spawns at `|x|>0.11`. It fails **identically on the clean tree**
(changes stashed, base `8da74b8a`) under mujoco 3.10.0/numpy 2.4.6 — a seed-brittle spawn-distribution assertion,
unrelated to R11.7A. Flagged, not fixed (out of scope).

## O0 parity gate — PASSED (three independent proofs)

1. **Model + collision + rollout bit-exactness** (`r11_7a_o0_parity.py`, production smoke, 3 demos): on the same
   reconstructed demo, Way A (direct `reconstruct_handoff(coin_shape="cylinder", disk_radius_override=0.020, …)`,
   unchanged = pre-refactor) vs Way B (`build_manipulation_rig(object_spec=COIN_OBJECT, robot=BALLTIP)`) →
   `model_diffs=none` (manipuland geom type/size, mass, nq/nv/nu/ngeom/nbody), **full collision contract**
   (`geom_contype`/`geom_conaffinity`) identical, deterministic rollout `max_dev=0.00e+00` on all 3.
2. **End-to-end certification:** the R11.6C `acquire_snapshot` at the frozen s1 seed (14250), through the **unpinned**
   `_make_env`, certifies (`certified=True`, `straddle0=−0.9962`, `n_dot=−0.997`).
3. **Provenance-hash reproduction:** the physical-state hash `16778d7df544b9e8` matches the committed frozen reference
   across 5+ artifacts (`bv_identification.json`, `teacher_bank.json`, `cradle_scout.json`, `dataset_contract.json`, the
   H2 identification report). The certified pipeline is unperturbed.

**Verdict:** `O0_PARITY_PASS` — the generated path is bit-identical to frozen R11.6C on the reference coin. Re-run and
**re-confirmed bit-exact after the full `ObjectSpec` threading (U3b)**. Per the gate, O1–O4 results are now unblocked.

## U3b / U5 (post-gate consolidation)

**U3b** threaded `ObjectSpec` as a single carrier through `reconstruct_handoff → replay_pi0 → CoinRL4Dof → neutral_env →
make_coin_env` (default `None` ⇒ back-compat loose kwargs ⇒ frozen coin); `build_manipulation_rig` now passes the spec
directly and the density/friction guard is removed (gap closed). Endpoint test proves a box+density spec yields a box
geom with elevated mass. `CoinStraddleTargets.for_object(footprint)` makes the straddle standoff footprint-aware
(`footprint + fingertip 0.02 + margin 0.015`; coin ⇒ 0.055 exactly). O0 parity re-run: all 3 demos bit-exact.

**U5** added a duplicate-ownership guard (`test_object_ownership_guard.py`): an AST scan asserts `compose_planar_scene`
has no new direct callers beyond the sanctioned set (canonical `planar_grasp_env.py` + 2 pinned pre-R11.7A direct
builders `coin_robot_variant.py` / `scenarios/kinematic_variant.py`, which may not grow), the allowlist has no stale
entries, and the bv `_make_env` object stays scene-sourced (no re-hardcoded literal). Loose object kwargs are marked
deprecated. **No deletions** (deferred to post-U6 per user decision).

## Performance

Peak RSS 249 MB (0.25 GB) for the parity smoke — ≪ the 16 GB cap. O0 parity regression 21.8 s (single frozen seed;
marked `slow`). No performance-sensitive path was modified (the generator is a thin wrapper; the model builder,
collision contract, and delivery rollout are untouched).

## Provenance

Git SHA `8da74b8a` (working tree dirty: the 4 modified + 6 new files above). Env: Python 3.11.15, mujoco 3.10.0,
numpy 2.4.6, macOS (Darwin 25.5.0, Apple Silicon), venv `hymeko_framework_rust/.venv`. Seeds: parity probe range
20000–20240 (demos at seed 20000, prefixes 67–69); frozen cradle 14250. Deterministic (fixed rng 20260806 for the
rollout actions).

## Open items / next stages

- **U3b** ✓ done (density/friction single-carrier threading + footprint-aware straddle).
- **U5** ✓ done (duplicate-ownership guard + deprecation; no deletions).
- **O3 (ellipse/capsule)** — still needs one new geom branch in `compose_planar_scene` (beside cylinder/box/triangle);
  the `Shape` enum has `CYLINDER`/`BOX`/`TRIANGLE` — an `ELLIPSE`/`CAPSULE` member + the builder branch is a U6 prereq.
- **Legacy consolidation (post-U6)** — the 2 pinned direct builders (`coin_robot_variant.py`,
  `scenarios/kinematic_variant.py`) and the deprecated loose object kwargs are removed only once O1–O4 have run green.
- **U6** — OBJ_O1–O4 as HyMeKo object declarations run through the exact-zero pipeline, with the stage-resolving
  failure taxonomy (`REACH_GEOMETRY` / `CAPTURE_PROPOSAL_TRANSFER` / `CONTACT_RETENTION` / `DELIVERY_POLICY_TRANSFER` /
  `TARGET_ENTRY`), a per-object certificate, and the R11.7A gate. **This is a compute-bearing campaign** (per-object bank
  generation) — scope (which families first, seed count) to be confirmed.
