# FANUC pick-place gripper↔table approach collision — investigation & corrected fix direction (2026-07-06)

**Author:** Aiko (agent) · verified locally (MuJoCo 3.9 + built CLI; kato15 was offline) · **Outcome:** the
attempted physics-mask fix BREAKS the grasp and was **reverted**; the real fix is a trajectory/geometry change.

## Problem

"The gripper still collides with the table and the object; during approaching, the gripper itself should not
collide."

## What was tried and what it showed (discriminating tests)

1. **Diagnosis:** the pick scene has NO collision filtering — every arm/gripper/floor/table/object geom is on
   MuJoCo's default `1/1`, and the only guard is a *reward penalty* (`approach_contact`, suppressed when over
   the object). Physics still resolves every gripper↔table / gripper↔object contact.
2. **Attempted fix — physics-mask (declarative `PickCollision` channels):** arm/palm isolated, fingers→object
   only, object rests on table, gripper↔table masked. Channel math verified correct.
3. **Result — it BREAKS the grasp.** Isolated A/B (local, expert rollout, n≥8):
   - default (no mask): **lift 1.0 / place 0.875** (matches kato15's 0.958/0.875 — local is a valid test env).
   - full mask: **lift 0.0 / place 0.0**; `ever_both_contact = False` — the fingers never grip.
   - Contact dump at closest approach: tool z **below** the object centre, **4 contacts all `world↔object`,
     0 finger↔object**. Cause: **the grasp relies on the table as a mechanical stop.** With gripper↔table
     masked, the descent overshoots — the fingers punch *through* the table and land below the object, so they
     never close on its sides.
4. **Where the approach collision actually comes from (the decisive measurement):** over 1561 approach steps
   (tool >6 cm from the object):
   - **finger↔table: 48.5%**  ·  **arm/palm↔table: 0.0%**  ·  grasp-phase finger↔table: 4 steps (negligible).

   **The relevant object is the gripper/fingers, so this means the approach is NOT clean.** The fingers — part
   of the gripper — collide with the table edge in 48.5% of approach steps; the arm/palm being 0.0% does **not**
   make the approach clean. Framing it as "arm/palm clean" would be wrong: if the fingers collide with the table
   edge on approach, the approach is not clean. (Masking arm/palm is grasp-safe but irrelevant; masking the
   fingers removes the collision but breaks the grasp — because the trajectory has no clearance of its own.)

## Primary diagnosis (this is the headline, not the mask result)

**The expert trajectory has a clearance problem.** The current expert succeeds only *partly*, and it does so by
**exploiting finger↔table contact as a mechanical stop** rather than maintaining clearance. A gripper/table-edge
collision during approach is **not an acceptable clean pick-place trajectory** — and since the gripper (its
fingers) is the relevant object, the fact that the arm/palm happen not to touch does not make the approach clean.

## 1. Collision masking — negative result, reverted, NOT the right fix

Verified above: default place 0.875 → full gripper↔table mask 0.0 (the grasp needs the table as a stop). The
`PickCollision` channel scheme + `_apply_pick_collision_channels` + its test were **reverted**; env restored
(place 0.833 / lift 1.0, ruff clean); kato15 unaffected (its sync failed while it was offline). **Do not
re-attempt collision masking.**

## 2. The actual problem

The gripper/finger **approach trajectory collides with the table edge** (finger↔table on 48.5% of approach
steps). The current expert **exploits that contact** (uses the table as a stop) instead of holding clearance,
so the demos it produces bake in a dirty, contact-exploiting approach — not a clean pick-place motion.

## 3. Future fix — a NEW VERSIONED experiment (next session)

1. **Modify the expert approach/transit trajectory** to clear the table edge: raise the transit/approach height
   (`hover_z`/`lift_clear`, pick_place_env.py:303-304; `_GRASP_TOOL_DZ` :284) and keep the wrist vertical earlier,
   descending only once the tool is above the object.
2. **Add a clearance check** for finger↔table AND gripper↔table contacts during approach — assert approach-phase
   gripper↔table → ~0 as a gate on the new trajectory (the clearance is the acceptance criterion, not just the
   grasp rate).
3. **Regenerate the demos** from the corrected, clearance-maintaining expert.
4. **Remeasure the expert ceiling** (place / lift, `LiftPlaceMetric`) on the clean trajectory — the prior
   0.875 ceiling was on the contact-exploiting expert and does not carry over.
5. **Rerun BC and DAgger as a new versioned experiment.** The prior pick-place BC (~0.55) and DAgger
   (best-ckpt ~0.79–0.83) numbers were produced on the contact-exploiting trajectory and **must not be conflated**
   with the clean-trajectory results; version the env/expert and label the new runs distinctly.
