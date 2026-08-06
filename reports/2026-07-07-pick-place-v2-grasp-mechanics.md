# v2 grasp-mechanics diagnosis + minimal fix (2026-07-07)

## Summary

Diagnosed why the near-object seeds grasp then lose the object, and applied the minimal grasp-logic fixes from the
allowed list (**latch the commit**, **v1's proven lift mechanics**, **re-centre the lift over the object**).
**Gains frozen; v1 byte-identical; clean transit preserved.** The fixes improve grip retention but **lift/place stay
0** — the diagnosis shows the object **slips/squirts out sideways** from an **off-centre grasp**, and that offset is
a **physical tracking offset (gains-limited, frozen)** — the same category as the far-object convergence.

## Diagnosis (near seeds 50000, 50003)

- They reach over-object early (~168/171), descend, close, and grasp with **both fingers** (grasp_hold→12, commit).
- **The grasp is ~2 cm OFF-CENTRE:** the damped descent settles the tool ~2 cm off the object (off ≈ 0.020–0.026),
  so the fingers straddle the box asymmetrically.
- **During the lift the object is DRAGGED SIDEWAYS and squirts out:** object xy drifts ~4 cm (e.g. −0.138→−0.100),
  the right finger drops (R→0), then both are lost; the object barely rises (objz 0.140→0.148, **not** lifted), and
  the tool climbs only ~4 cm over 100+ steps (the IK toward the lift target folds at the reach edge).
- **Which of slip / open / no-command?** → **the object SLIPS** (squirts out). The gripper does **not** open (the
  commit latch keeps it closed), and the lift command **is** issued.

## Fixes applied (all logic-only, clearance-safe, supported by the diagnosis)

1. **Latch the commit** (allowed #7): once both fingers hold for the dwell, stay committed through a momentary
   contact flicker; release only if the object is clearly dropped (fell below the surface). Stops the
   CARRY→un-commit→open-grip oscillation.
2. **v1's proven lift/carry mechanics** (allowed #4/#6): CARRY uses `_ik_step` (rate 0.28), not the slow
   `_v2_ik_step` (0.10) toward an above-ceiling target that barely lifted.
3. **Re-centre the lift over the OBJECT** (capture the object xy, not the off-centre tool xy) so lifting pulls the
   gripper back onto the box.

(The lift-target lowering was tried and reverted — it did not help; the box squirts out *before* the lift height
matters.)

## Results (v2 grasp-mechanics smoke, 4 ep)

- forbidden-pre-object **0.0** · transit finger↔table **0.0** · physical min clearance **0.0** (grazing) —
  **clean transit preserved** (crit1/crit2 pass, no regression).
- **grasp rate 2/4** (near seeds grasp both fingers; far seeds don't reach over-object in time).
- **lift rate 0/4 · place rate 0/4.**
- near-seed **object z after lift command ≈ 0.145–0.148** (barely above the resting 0.140; needs > 0.175 to count).
- **grasp retained through lift? NO** — the object squirts out and the fingers are lost.
- gate: **FAIL** (crit1+crit2 pass; crit3 grazing; lift/place 0).

## Checklist answers

- **Changed files:** `hymeko_rl/env/pick_place_env.py` only.
- **Gains changed?** No (kp60/kv15/dt÷2 frozen).
- **v1 behavior changed?** No — byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577); 15 tests green.
- **Clean-transit metrics:** forbidden 0.0, transit 0.0, clearance non-negative — preserved.
- **grasp / lift / place:** 2/4 · 0/4 · 0/4.
- **Next blocker:** **grasp centring** for the near seeds — and it is **physical tracking (gains-limited)**, the
  same underlying limit as the far-object convergence. Not a clearance-margin issue, not a place issue.

## Verdict (per your decision rule)

- *"near seeds still grasp but do not lift → inspect slip / open / no-command"* → **the object SLIPS** (squirts out
  from the off-centre grasp). Done.
- *"clean transit regresses → revert"* → it did **not** regress; kept the fixes.
- *"far seeds fail due to convergence → keep as a separate gain/reach-limit issue"* → yes.

**Conclusion:** the v2 clean-transit expert is achieved (crit1/crit2 pass), but **both** remaining blockers —
grasp **centring** (near) and over-object **convergence** (far) — are the **same physical tracking/gains
limitation**, which is **frozen** per your instruction. Further lift/place progress needs the gain/tracking lever
(or a scene change), both out of scope here. The grasp-logic fixes are kept (they succeed the moment the tracking
offset is small enough). Stopped after the smoke; 32-ep gate not run. No BC/DAgger/RL.
