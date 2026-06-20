# hymeko_rl Phase 0 — the kinematic-hypergraph bridge + HSiKAN policy reads it

**Date:** 2026-06-18
**Plan:** [docs/plans/2026-06-18-mujoco-rl-grasping](../docs/plans/2026-06-18-mujoco-rl-grasping/).
**Status:** ✅ the obs-hypergraph bridge + the HSiKAN/Gömb backbone are wired and tested;
`build_policy("hsikan"|"mlp")` is the architecture-ablation switch. The distinctive claim
— *the policy reads the compiled hypergraph, not raw joints* — is demonstrated end-to-end
at the network level. Two upstream framework bugs found and worked around (below).

## Summary
Realised Csaba's pipeline (hymeko → mjcf, *then* the observation as a hypergraph with an
equivalent tensor): `HypergraphState` extracts the signed kinematic hypergraph from the
compiled MJCF — vertices = links, hyperedges = joints signed by parent(+)/child(−)
direction — and `HSiKANBackbone` does dense, batched signed message passing over it,
feeding the shared `ActorCritic` actor+critic heads. The MLP baseline reads a flat obs;
the only difference is the backbone, so the planned HSiKAN-vs-MLP comparison is a
`--policy {hsikan,mlp}` switch.

## Two upstream bugs found (worked around, tracked)
1. **PyO3 engine `load_file` import resolver** — robot `.hymeko` files
   (`@"meta_kinematics.hymeko"`) fail with `file not found` even when the sibling exists
   (cwd-independent); the **CLI resolves imports correctly** (it emits the MJCF). So the
   canonical `.hymeko` star-expansion (engine `compile_star_expansion`) is unreachable via
   the binding for imported robots.
2. **Default `hymeko emit -f mjcf`** produces an invalid actuator (`unknown transmission
   target 'j0'`) — the emitted MJCF won't load in MuJoCo. The **`*_faithful.mjcf`** emit
   (a different path) is clean and loads (6-DOF).

**Work-around (faithful to the plan):** the bridge consumes the working *faithful* MJCF
via `from_mjcf` (robust, dependency-free). `from_hymeko` (CLI emit → `from_mjcf`) is
implemented and correct but blocked by bug 2 on the default emitter; it works the moment a
clean MJCF is emitted. The MJCF-derived hypergraph is structurally faithful (same compiled
robot, parent/child joint directionality); the engine-direct star-expansion is a tracked
refinement once bug 1 is fixed.

## Files touched
**New:**
- `hymeko_rl/hypergraph_state.py` (+150) — `HypergraphState` (`from_mjcf`, `from_hymeko`
  via CLI, `dense_signed_adj` row-normalised, `topo_hash` re-encode gate).
- `hymeko_rl/tests/test_hypergraph_state.py` (+75) — bridge counts/signs, adjacency
  normalisation, hash stability, and the HSiKAN policy reading `(B, N, in_feat)`.
**Modified (mine):**
- `hymeko_rl/policy.py` — `_SignedConv` + `HSiKANBackbone` (batched dense signed message
  passing; the sparse transductive sibling is `signedkan_wip` `SGCNLayer`) + `hsikan_backbone`;
  registered `hsikan` in the backbone registry; `build_policy("hsikan", …, hg_state=…)` now
  works (the old `NotImplementedError` is gone). `policy` stays mujoco-free (`TYPE_CHECKING`
  import of `HypergraphState`, duck-typed at runtime).
- `hymeko_rl/tests/test_policy.py` — the `hsikan` error test now asserts `TypeError`
  (missing mandatory `hg_state`) instead of `NotImplementedError`.

**CORE.YAML:** none. **No new dependency.**

## Test results
- `hymeko_rl/tests/` **17 passed** (7.0 s, `pytest -p no:randomly`): the 12 prior +
  5 new (bridge + HSiKAN policy). `ruff check` clean; `mypy --strict` clean on the two
  touched modules.

## §6.5 anti-patterns
None. Backbone Strategy + registry (one `build_policy`, no per-kind wrappers, §6.5 #1/#9);
`ActorCritic` serves actor+critic with no duplication; the signed conv is a small batched
variant of the existing SGCN idea (noted, not a blind re-impl, §6.5 #2); errors are
specific (`ValueError`/`TypeError`, §6.4); no globals (the registry is an immutable map of
pure callables).

## Performance
Trivial: 17 tests in 7 s; the arm hypergraph is 7 vertices / 12 arcs, dense adjacency
`(7,7)`. Peak RSS ≪ 16 GB. No training yet (no benchmark claim).

## Provenance
Git SHA `7d16ad0` (tree dirty). `mujoco 3.9.0`, `gymnasium 1.3.0`, torch 2.12.0+cu132.
Arm MJCF: `demos/hero/out/anthropomorphic_arm_faithful.mjcf` (compiled from
`data/robotics/anthropomorphic_arm.hymeko`). Seeds fixed in tests.

## Open / next
1. **Phase 1 — REACHING (BC):** `ArmEnv` (Gymnasium) with per-vertex obs on the kinematic
   hypergraph; behaviour-clone from a scripted demonstrator; first shown milestone.
2. **File the two upstream bugs** (engine import resolver; default mjcf-emit actuator) —
   fixing bug 1 unlocks the canonical star-expansion path; bug 2 unlocks `from_hymeko`.
3. **Quadruped** reuses this bridge unchanged (its branched topology is where the
   HSiKAN-vs-MLP gap should be largest).
