# FANUC pick-place `expert_version=2` — deterministic clearance-aware waypoint controller (2026-07-07)

## Summary

Implemented `expert_version=2` as an 8-segment deterministic waypoint state machine
(`HOME_SAFE_RISE → TRANSIT_ABOVE_TABLE → ABOVE_OBJECT_ALIGN → VERTICAL_DESCENT → GRASP → LIFT → PLACE_TRANSIT →
PLACE_DESCEND_RELEASE`, plan bundle `docs/plans/2026-07-07-pick-place-v2-waypoint-planner/`) with the two design
levers: **short Cartesian hops** and **IK seeding from the previous un-folded config** (a commanded-config chain
from `arm_home`, via a new `DampedPoseIK.fk_tool`). v1 is preserved **byte-for-byte** (proven below).

**The v2 smoke does not pass the gate.** A discriminating probe **overturned** my first (premature) "geometry
wall" hypothesis and isolated the real blocker: a **retraction stall from the extended home pose**, not an
unreachable object.

## Files changed

| file | change | v1 impact |
|---|---|---|
| `hymeko_rl/env/pick_place_env.py` | rewrote `_expert_action_v2` as the 8-segment FSM; added `_v2_ik_step` (seed-frame hop + seed chain); added `_v2_seed` field (init + reset). `_expert_action_v1` and `_ik_step` **untouched**. | none (byte-identical) |
| `hymeko_rl/env/ik.py` | added `DampedPoseIK.fk_tool(q)` (scratch-data FK; reusable). | none (v1 does not call it) |
| `hymeko_rl/tests/test_ik.py` | +`test_fk_tool_matches_kinematics`. | — |
| `hymeko_rl/tests/test_fanuc_pick.py` | +`test_v2_expert_path_runs_and_is_deterministic`, +`test_v2_preserves_v1_default_and_dispatch`. | — |

**No reward, BC, DAgger, RL, coin-collab v2b, or remote kato15 change.** CORE.YAML: none touched (verified
`pick_place_env.py`/`ik.py` are not core-listed).

## v1 regression check — PASS (byte-identical)

`v1_clearance_resmoke` (4 ep, seeds 50000–50003) reproduced the earlier v1 signature **exactly**: seed 50000 lift 1
place 1 first_fingtab 51 first_over 225 min_clr −0.02577 transit 0.4464; aggregate lift 1.0 / place 0.75. The
`reset()`/`__init__` additions and `fk_tool` did not perturb v1. `test_v2_preserves_v1_default_and_dispatch` also
guards the default + dispatch.

## v2 smoke — FAIL (does not pass the gate)

`v2_clearance_smoke` (4 ep, seeds 50000–50003, horizon 620):

| metric | v2 | v1 (ref) |
|---|---|---|
| lift / place | **0.0 / 0.0** | 1.0 / 0.75 |
| forbidden-pre-object rate | **1.0** | 1.0 |
| transit finger↔table rate | **0.94** | 0.45 |
| min clearance (m) | **−0.028 … −0.018** | −0.026 |
| first finger↔table strike | ~37 | ~51 |
| first over-object | **None (never)** | ~149–225 |

Gate: crit1/2/3 all **False** → **FAIL**.

## Root cause (discriminating probe, seed 50000) — NOT a geometry wall

- Object at r = **0.306** (reachable — v1 grasps it); `arm_home` places the tool at r = **0.669** (very extended).
- The commanded config **retracts inward** from r 0.669 → r 0.498 in ~40 steps, then **stalls** at
  (0.49, −0.09, cmd z 0.305) for the remaining ~580 steps — it never reaches the object's r 0.306. The DLS cannot
  retract further inward at constant `z_hover` while holding the tool down.
- The **physical wrist sags ~8 cm below the commanded z** (0.225 vs 0.305) at that extension → the fingers drag the
  table (transit contact 0.94, negative clearance).
- **Measured vs inferred:** *measured* — the stall radius, the sag, first_over=None on all 4 seeds including the
  reachable-radius ones. *Inferred* — the DLS stalls because inward retraction at fixed z with the down-constraint
  is near-singular from the extended seed; the sag is insufficient servo authority at full extension. *Overturned*
  — the earlier "z_hover unreachable over the far object" story (it fails even at r 0.306, which is reachable).
- v1 reaches the object (first_over ~225) **only via the fold/dip that makes it dirty** — the exact motion the v2
  anti-fold levers suppress. So the clean levers and reaching-the-inward-object are, as implemented, in tension.

## Is the 32-episode gate justified? — NO

The 4-episode smoke fails all three hard criteria and the controller never reaches the object. Running the 32-ep
gate now would only re-measure a known failure at 8× cost. **Not justified** until the retraction stall is fixed.

## Recommended next levers (user decision — not applied; each needs a re-smoke)

1. **Retracted seed / multi-start solve** — seed `_v2_ik_step` from a retracted high-elbow config, or swap
   `solve` → the existing `solve_collision_free` (multi-start, proven by `test_fanuc_top_down_grasp_pose_is_collision_free`)
   to escape the inward-retraction stall. (Per-step cost rises; measure.)
2. **Retract-first waypoint ordering** — an explicit early segment that pulls the tool inward toward the base at
   high z *before* transiting to the object, so no segment asks for inward-at-constant-z from full extension.
3. **Tracking authority** — the ~8 cm wrist sag at extension suggests the position gain / gravity-comp is
   insufficient there; stronger arm gains would reduce the drag (a control-tuning change, measure both axes).
4. **Fallback (design-note): a versioned scene/object change** (closer `obj_radius`, less-extended `arm_home`) —
   `arm_home` is shared with v1, so any change is versioned and gated on your decision.

I stopped after the v2 smoke per the task bound; picking among 1–4 is a design decision for you.

## Tests / gates

- ruff (env, ik, tests): clean · mypy --strict (ik.py): clean.
- pytest ik + fanuc_pick + pick_place_env + pick_clearance: **23 passed**.
- v1 re-smoke: byte-identical (no regression). v2 smoke: FAIL (documented above).
