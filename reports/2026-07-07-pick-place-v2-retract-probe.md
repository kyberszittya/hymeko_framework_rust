# v2 retracted-seed feasibility probe — is HOME_RETRACT_OR_PRESHAPE justified? (2026-07-07)

## Summary

Froze the failed v2 smoke as a labeled negative result and built a **bounded kinematic feasibility probe**
(`hymeko_rl.eval.pick_retract_probe`) to answer: does a clearance-safe transition exist from the over-extended
`arm_home` into a usable retracted branch from which the object hover is reachable? **Verdict: FEASIBLE on 3/3
seeds.** The probe also **overturned the "geometry wall"** framing for good and isolated the true blocker — a
physical wrist sag mis-firing a controller branch, not kinematic reachability.

No physics stepping, no gains, no scene change, no 32-ep gate, no BC/DAgger.

## Frozen negative result

`reports/figures/pick_place_clean_expert/v2_waypoint_controller_smoke_failed_retraction_stall.{json,csv,png}`
(copy of the failed 4-ep v2 smoke: lift/place 0/0, first_over=None, transit contact 0.94). Memory
`project-pick-place-v2-retraction-stall`.

## Probe design (kinematic, reuses `solve` / `solve_collision_free` / `fk_tool`)

Per candidate seed `q`: FK tool pose; signed **finger↔table** clearance (`mj_geomDistance`); self/floor-collision
status (reported, not gated); table-clear **home→seed** joint interpolation; table-clear **seed→hover** short-hop
chain (the `_v2_ik_step` rule). Candidates: `arm_home`, `cf_hover` (multi-start collision-free at the object
hover), `cf_mid_retract` (multi-start at a retracted mid-radius pose), `manual_elbow_up` (hand guess).

## Probe self-correction (the probe was initially wrong — verified before concluding)

First run returned "infeasible everywhere" — an **artifact**, not the answer. A diagnostic showed `arm_home` trips
`_ik_pose_valid` only because of a **benign 3 mm `link_2↔link_4` self-overlap** baked into the home posture (v1
runs from it). That poisoned step 0 of every path, and treating a 0.0 graze as forbidden compounded it. The user's
criterion is explicitly "collision-free **or at least table-clear**", so the gate was corrected to **finger↔table
penetration (`clr < 0`)**; self-collision is reported, not gated. Re-run → clean, consistent verdict.

## Results (corrected predicate)

| seed | obj r | best seed | `cf_hover`: seed clr / cf / reaches hover | `cf_mid_retract` clr | verdict |
|---|---|---|---|---|---|
| 50000 | 0.306 | cf_hover | **+0.063 / yes / yes** | 0.0 | **FEASIBLE** |
| 50001 | 0.393 | cf_hover | **+0.064 / yes / yes** | +0.124 | **FEASIBLE** |
| 50002 | 0.379 | cf_mid_retract | +0.064 / yes / yes | +0.124 | **FEASIBLE** |

- A **table-clear (+6.3 cm), collision-free config sits exactly at the object hover** (`cf_hover`) for **every**
  seed — including the far **r=0.393** object. So `z_hover=0.305` **is** cleanly reachable over the far object; the
  earlier "unreachable hover" hypothesis is fully dispelled.
- A retracted intermediate (`cf_mid_retract`, r≈0.17–0.22, +0.12 m clearance) is also clean and reaches the hover.
- The joint-space `home→seed` transition does **not penetrate** the table (min clearance 0.0 graze → +0.21).

## The true blocker (bonus finding, correlating the probe with the earlier physical trace)

The probe's kinematic command chain **reaches the hover even from `arm_home`** — yet the real controller stalled.
The difference is **physical**: the wrist **sags ~8 cm** below the commanded z at extension (0.225 vs 0.305), and
that sagged *physical* z drops below `z_hover − 0.02`, which **mis-fires the `HOME_SAFE_RISE` branch** (hold-xy,
rise) — halting lateral transit and locking the stall. So the v2 failure is **sag + a sag-sensitive phase
condition, not kinematic reachability.**

## Verdict — HOME_RETRACT_OR_PRESHAPE is feasible

**Justified (3/3).** A concrete clean target exists per seed (multi-start `cf_hover`, +6 cm clearance), reachable
from `arm_home` by a table-clear joint-space move. Recommended shape for the (next) phase: a **joint-space preshape
to a multi-start collision-free seed** at/above the object hover, *then* the Cartesian descent — rather than the
current Cartesian transit from full extension.

**Coupled caveat (must be addressed for the controller to realise the feasible path):** the phase decision must not
key off the **sagged physical z** (base `HOME_SAFE_RISE` on the commanded/seed z, or drop the low-z RISE trigger
during transit); and the ~8 cm sag itself is a tracking-authority issue (gains) **deferred per your instruction**.
A retract phase alone, layered on the current sag-sensitive FSM, may not suffice without this.

## Files / tests / gates

- Added `hymeko_rl/eval/pick_retract_probe.py` (+`test_pick_retract_probe.py`, 2 tests).
- Froze the negative smoke artifacts under `v2_waypoint_controller_smoke_failed_retraction_stall.*`.
- No change to `pick_place_env.py`, `ik.py`, reward, BC/DAgger, coin-collab v2b, or kato15.
- ruff clean · mypy --strict (probe) clean · pytest 2/2. Probe artifact:
  `reports/figures/pick_place_clean_expert/retract_probe.json`.

**Stopped after the probe** (per instruction). `HOME_RETRACT_OR_PRESHAPE` is **feasible** — awaiting your go to
implement it, with the sag/phase-condition caveat above.
