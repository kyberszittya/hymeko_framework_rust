# v2 clearance-aware FANUC pick expert — waypoint-planner DESIGN NOTE (design-only)

**Status:** design note only. **Ownership:** the parallel track owns the v2 *implementation* (`expert_version=2`,
`_expert_action_v2`); this thread contributes design/diagnosis and does **not** edit v2 code. **Not an
overturning of the v2 freeze — a refinement of its cause.** No BC/DAgger until the expert passes the clearance
gate.

## Refined diagnosis (recorded)

- Static top-down IK poses **with clearance do exist**.
- `grasp_z + 0.06` (z ≈ 0.315) is IK-valid and clears the table (fingers ~7.5 cm above the top).
- Higher hover targets (`+0.10 / +0.14`) can **self-collide** or leave the useful envelope.
- Therefore the problem is **not** simply static reachability.
- The blocker is the **rate-limited DLS-IK trajectory path** from `arm_home` to the hover target.
- That path can **dip below the table en route** (measured: min clearance −0.11 m targeting the +0.06 hover),
  even when the final target pose is valid.
- Therefore a **hover-height tweak is insufficient** — the fix must control the **path**.

## Why the direct DLS dips (mechanism)

`env._ik.solve(q_now, far_target, down=True)` takes the DLS "short path" in *joint* space toward a far target;
crucially it solves from the **drifting `q_now`** each step (120 iters, rate 0.22), so from `arm_home` the arm
folds low and the tool descends into a folded local minimum before it reaches the target xy — the fingers cross
the table plane mid-motion (tool starts ~0.387, sinks to ~0.137, horiz freezes ~0.22, never centres). Rate-limiting
bounds the *step* size but not the *direction*. Static reachability (a fresh IK from HOME reaches `(obj, 0.315)`)
≠ the closed-loop trajectory. The parallel track instrument-confirmed the same: the blocker is **IK seeding /
closed-loop path**, not the envelope.

**Two levers fix this (either or both); height is already solved:**
- **(A) IK seeding** — seed each solve from a **good, un-folded config** (home / high-elbow), or blend `q_now`
  toward it, instead of the drifting `q_now`. Keeps the solver out of the folded basin.
- **(B) Short Cartesian waypoints** — make every DLS solve a **short hop** whose target is close to the current
  tool pose, so the solver never has to fold: sub-waypoint the Cartesian path (rise → level traverse → descend),
  each intermediate pose inside the valid reachable envelope.

The waypoint plan below is lever (B); it composes with (A) (seed each hop from a good config).

## Correct v2 design direction — explicit waypoint plan

Replace the single far-target DLS with a staged Cartesian path; each stage targets the **next waypoint** (a small
Cartesian step from the current tool pose), not the far endpoint. Invariant to hold at **every** step: the lowest
finger geometry stays **above the table top by ≥ a clearance margin** until the tool is directly over the object.

1. **Rise from home** — straight up (hold xy, increase z) to a safe transit height `z_transit` that is inside the
   valid envelope everywhere the traverse will pass (see constraints).
2. **Lateral traverse with clearance** — move xy toward the object **at `z_transit`** (hold z), in small hops, so
   the tool tracks a level path and never dips. Do not target `grasp_z` while off-centre.
3. **Enter the object region from a valid envelope** — arrive above the object still at `z_transit` (finger
   clearance positive).
4. **Vertical descent only above the object** — with the tool centred (`horiz ≤ ~0.02`), descend straight down
   (hold xy, decrease z) from `z_transit` to `grasp_z`, damped, fingers WIDE.
5. **Grasp** — at `grasp_z`, close; dwell to settle (`_grasp_hold`).
6. **Lift** — straight up from the captured grasp xy to a lift-clear height.
7. **Place** — traverse at lift-clear to the target, lower, release.

## Design constraints (measured 2026-07-06)

- **Valid clearance-hover ceiling ≈ `grasp_z + 0.06`** over the object (z ≈ 0.315). Above → self-collision. So
  `z_transit` over the object is bounded by ~grasp_z + 0.06.
- **`z_transit` need not be constant** — the reachable top-down height varies with radius. Prefer
  `z_transit(r) = min(reach_ceiling(r) − margin, z_that_clears_table_edge)`. If a single level height cannot both
  clear the table edge on the outward radius *and* stay reachable over the object, use a **radius-scheduled**
  height (rise higher near the base where the envelope is generous, descend to the reachable hover over the
  object) — always as short hops.
- **Small hops** (Cartesian step ≤ a few cm) are the load-bearing detail — they keep each DLS solve un-folded.
  Rate-limit per hop as today, but on a *near* target.
- **Fallback (only if path planning still cannot clear):** a versioned scene/object change (closer `obj_radius`,
  lower `table_top` → lower `grasp_z` vs the reach ceiling). **Not forced** — valid clearance poses exist, so try
  the waypoint path first.

## Acceptance gate (UNCHANGED)

Run the expert-only clearance harness (N=24/32) and require **all**:
- no table-edge strike before the gripper is above the object,
- transit finger↔table contact = 0 or near-zero,
- minimum vertical clearance **positive** during transit/approach,
- lift/place remains reasonably high (target place ≥ 0.80, lift ≥ 0.90; clearance is mandatory — do not accept a
  high-success dirty trajectory).

**Only after the expert passes the gate:** regenerate demos (v2), recompute the expert ceiling (v2), train BC v2,
run declarative DAgger v2 (`algorithm "dagger";` TrainingSpec path), label everything
`pick_place_v2_clearance_aware`, and **do not overwrite** the `v1_dirty` numbers.

## Validation apparatus already available

- `_ik_step` (shared IK helper), the `expert_version` param, `_expert_action_v2` scaffold + `_V2_*` knobs (parallel
  track).
- Clearance harness pattern: per-step finger/gripper↔table contacts, table-edge strikes, signed min clearance
  (lowest finger box-corner z vs table top), first-contact vs first-over-object timing, lift/place. (See
  `reports/2026-07-06-pick-place-clearance-forensics.md` for the v1 baseline and the metric definitions.)

## Coordination

Parallel track owns v2 implementation. This thread: design/diagnosis only — no v2 code edits unless explicitly
handed the implementation.
