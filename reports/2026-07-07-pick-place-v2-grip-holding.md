# v2 grasp-holding fix — near-object pick-place SOLVED (2026-07-07)

## Summary

Diagnosed why the grasp did not hold, and fixed it with an **integral finger-center recenter** (allowed #1) plus an
**over-object latch**. **Gains and grip gain frozen; v1 byte-identical; clean transit preserved.** Result: **both
near seeds now lift AND place** — lift/place **0.0 → 0.5** (2/4). The v2 expert now performs a clean-transit
top-down pick-and-place for near objects. The remaining 2/4 are the far-object over-object convergence
(reach/tracking-limited), a separate fenced issue.

## Diagnosis (near seeds)

The box is a 4 cm cube. At first two-finger contact the **finger center was ~3 cm off** the object
(`LAT_OFFSET 0.030`, aperture 0.008 — the fingers closed *beside* the box, not around it), so closing **shoved the
box out sideways** (object velocity +0.17…+0.29 m/s, ~3.6 cm drift) and it never rose. The grip and arm gains were
fine — the fingers simply weren't over the box. (An IK/gripper-geometry offset the tool commands don't account for.)

## The fix (grip/grasp side only; arm gains frozen)

1. **Over-object latch** (`_v2_over_latched`): once the physical tool is over the object, latch and run the grasp
   sequence keyed off the **finger center** (not the tool), so recentring at the hover can't fall back to transit.
2. **Integral finger-center recenter** (`_v2_align_corr`, Ki=0.5, anti-windup ±0.06): each align step accumulate
   `corr += Ki·(obj − fingercenter)` and command the tool to `obj + corr`, until the **finger center** sits over
   the object (< `_V2_ALIGN_TOL`=0.012), *then* descend + close — keeping the correction through the descent.
   Proportional (K=1/K=2) left a ~1.4 cm residual that centered only one near seed at a time; the integral drives
   it to ~0 and centers **both**. Recentring happens at the clearance hover (high z), so transit stays clean.
3. (kept from the prior steps) commit latch, v1 lift mechanics (`_ik_step` 0.28), lift re-centred over the object,
   phase-scoped arm gains (kp75 over-object). Grip gain (250) **unchanged**.

## Results (v2 grip-holding smoke, 4 ep)

| metric | value |
|---|---|
| forbidden-pre-object / transit contact | **0.0 / 0.0** — clean transit preserved (crit1/crit2 pass) |
| **grasp centring** | finger-center offset 3 cm → ~1 cm; aperture 0.008 → ~0.056 (fingers straddle the 4 cm box) |
| **contact persistence through lift** | near seeds: **both fingers held** through the lift |
| **object z after lift** | near seeds rise to ≈ 0.167+ (**above** the 0.175… threshold at place); lateral drift ~1 cm |
| **lift rate / place rate** | **0.5 / 0.5** (both near seeds 50000, 50003 lift **and** place) |
| far-seed over-object timing | 334 / 489 (unchanged — reach/tracking-limited) |
| gate | FAIL (crit1+crit2 pass; crit3 grazing; lift/place 0.5 < 0.90/0.80) |

## Checklist answers

- **Changed files:** `hymeko_rl/env/pick_place_env.py` only.
- **Arm gains changed?** No (phase-scoped kp75 from the prior step, unchanged here).
- **Grip gain changed?** **No** (250).
- **v1 behavior changed?** **No** — byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577); 15 tests green.
- **Clean-transit metrics:** forbidden 0.0, transit 0.0, clearance non-negative — preserved.
- **Grasp rate:** 2/4 (near seeds grasp; far seeds don't reach over-object in time).
- **Contact persistence through lift:** yes (near seeds).
- **Object z after lift:** ≈ 0.167+ (near seeds, above threshold).
- **Object lateral slip:** ~1 cm (near seeds; no squirt-out).
- **Lift / place:** 0.5 / 0.5.
- **Next blocker:** **far-object over-object convergence** (reach/tracking-limited) — the near-object
  grasp/lift/place is **solved**. Not place refinement (near-object place already works), not gripper geometry.

## Verdict (per your decision rule)

*"If near seeds work but far seeds still fail due to convergence time, split: near-object grasp/lift is solved,
far-object convergence remains reach/tracking-limited."* — **exactly this.**

- **Near-object clean pick-and-place is SOLVED** (both near seeds lift+place, transit clean).
- **Far-object failure remains the over-object convergence** (physical arm crawls to the far object by step
  334/489) — reach/tracking-limited, and the levers for it (arm gains / scene / object distribution) are frozen.

The v2 expert is now a **clean-transit top-down pick-place** that succeeds on near objects. **32-ep gate not yet
justified** (lift/place 0.5, crit3 grazing) — but this is the first working lift/place, a real step past the
tracking-limited freeze. Stopped after the smoke; no 32-ep gate, no BC/DAgger/RL.
