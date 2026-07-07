# v2 phase-scoped tracking/gain experiment (2026-07-07)

## Summary

Froze the current result as `v2_clean_transit_no_lift_tracking_limited`, then ran the bounded, v2-only,
**phase-scoped** tracking/gain experiment: transit keeps the clean kp60 base; a stronger arm gain applies **only
once over the object** (descent/grasp/carry). **Gains stay v2-only; v1 byte-identical; clean transit preserved; no
instability.** Result: **lift/place stay 0** — and the experiment isolates *why*: raising the **arm** gain lifts the
**tool**, not the box. The box slips out during the lift **regardless of arm gain** (rises ~0.008 m at kp60, kp75,
*and* kp90). So the lift bottleneck is **grip/grasp holding, not arm tracking.** Per your decision rule this is
"no lift even with clean tracking → **freeze v2 as clean-transit/no-lift and pivot.**"

## Exact gain/tracking changes (v2-only, phase-scoped)

- New runtime gain-setter `_v2_set_arm_gains(kp, kv)` (arm actuators only; grip untouched).
- Per-phase selection in `_expert_action_v2`: **transit** (preshape / hold-hover) → **kp60/kv15** (unchanged base);
  **over the object** (descent / grasp / carry, i.e. `committed or both or (preshaped and physical-over-object)`)
  → **kp75/kv18**.
- Deliberately **phase-scoped**, not the rejected GLOBAL kp90 (which knocked the object off). Also tested a
  phase-scoped **kp90/kv20** — stable (qacc ≈ 900, no knock-off) but no lift either.
- Grip gain (250) and dt÷2 unchanged. `__init__` base gains unchanged (kp60 v2 / kp45 v1).

## Results (v2 phase-scoped smoke, 4 ep) — kp75

| metric | value |
|---|---|
| forbidden-pre-object / transit contact | **0.0 / 0.0** — clean transit preserved (crit1/crit2 pass) |
| first strike | after over-object or **absent** (all four seeds `first_fingtab=None` or later) |
| grasp rate | 2/4 (near seeds grasp both fingers) |
| **near-seed grasp retention through lift** | **NO** — the box slips out |
| **object z after lift command** | **≈ 0.148** (rises only ~0.008 m; needs > 0.175) |
| **object lateral slip** | **YES** — drifts ~4 cm sideways, squirts out of the fingers |
| far-seed over-object timing | 334 / 489 (unchanged — still slow) |
| lift / place | **0.0 / 0.0** |
| instability / object knock-off | **none** (qacc ≈ 900, phase-scoped avoided the global-kp90 blow-up) |

**The decisive number:** object lifted vs arm gain — kp60 → 0.008 m, kp75 → 0.008 m, kp90 → 0.008 m. Arm gain
raises the **tool** (max tool z 0.28 → 0.32 → 0.336) but the **box does not follow**. The grasp does not hold.

## Checklist answers

- **Changed files:** `hymeko_rl/env/pick_place_env.py` only.
- **v1 behavior changed?** **No** — byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577); 15 tests green.
- **Clean-transit metrics:** forbidden 0.0, transit 0.0, clearance non-negative — preserved.
- **Near-seed grasp retention:** not retained (box slips).
- **Object z after lift:** ≈ 0.148 (≈ +0.008 m, below the 0.035 threshold).
- **Object lateral slip:** yes (~4 cm).
- **Far-seed over-object timing:** unchanged (334/489).
- **Lift / place:** 0.0 / 0.0.
- **Instability / knock-off:** none.
- **32-episode gate justified?** **No.**

## Decision — freeze v2 as clean-transit/no-lift and pivot

Per your rule: *"If no lift appears even with clean tracking, freeze v2 as clean-transit/no-lift and pivot."* That
is the case. The **arm** tracking/gain lever is exhausted — it is not the lift bottleneck; the box slips out of the
grip regardless of arm gain. The remaining blocker is **grasp holding** (grip force/geometry / the damped-descent
grasp quality) and, separately, the **far-object convergence** — both outside the arm-tracking lever and both
touching areas you've fenced off (scene / object distribution / reward / grip geometry).

**Frozen:** `reports/figures/pick_place_clean_expert/v2_clean_transit_no_lift_tracking_limited.{json,csv,png}` —
the v2 clean-transit expert (crit1+crit2 pass, forbidden 0, transit 0, clearance non-negative), lift/place 0,
tracking-limited.

**Suggested pivot options (your call — none applied):**
1. Grip/grasp-holding fix (grip gain / finger geometry / grasp-pose approach) — a *grasp*-side change, distinct
   from arm tracking.
2. A bounded scene tweak (e.g. slightly closer object radius) — resolves *both* the far-object convergence and
   likely the grasp — but you've fenced the scene for now.
3. Accept v2 as the clean-transit demonstrator and pivot the line (e.g. back to the standing coin-collab/other
   work), documenting that a clean-transit top-down pick is achieved but the lift tail needs a grasp/scene change.

Stopped after the smoke; 32-ep gate not run. No BC/DAgger/RL. Gains phase-scoped (kp75) and v2-only; v1 frozen.
