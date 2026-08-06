# v2 monotonic multi-waypoint preshape route (2026-07-07)

## Summary

Replaced the single-target Cartesian preshape (which oscillated progress-down/lift-up) with a **monotonic
multi-waypoint route**: home → **cf_mid_retract** (joint, slow, high, no descent) → **HOLD** (physical catch-up) →
short **capped Cartesian hops** down to **cf_hover** (z clamped ≥ floor, no lift-up). **Gains frozen**
(kp60/kv15/dt÷2); v1 byte-identical. **This makes the transit clean — crit1 and crit2 now PASS** (forbidden-pre-object
1.0→0.0, transit contact 0.38→0.0, transit clearance −0.008→0.0). The gate still fails: crit3 is grazing (0.0, not
strictly positive) and **lift/place = 0 — a grasp/descent problem, not a clearance one.**

## Changed files / gains / v1

- Changed: `hymeko_rl/env/pick_place_env.py` only (monotonic route: `_v2_preshape_waypoints` + `_v2_preshape_step`
  phases + fields/constants). **Gains unchanged** (kp60/kv15/dt÷2 frozen). **v1 byte-identical** (re-smoke exact:
  lift 1.0 / place 0.75 / min_clr −0.02577); 15 tests green; ruff clean.
- **Route:** `home → cf_mid_retract` (joint, rate 0.05) → **hold 20 steps** → `cf_mid → cf_hover` (Cartesian hops,
  xy cap 0.03, z cap 0.02, z clamped ≥ floor) → hold-hover → descend.

## Results (v2 smoke, 4 ep, seeds 50000–50003)

| metric | v2 single-target Cartesian | **v2 monotonic multi-waypoint** |
|---|---|---|
| forbidden-pre-object rate | 1.0 | **0.0** ✅ crit1 |
| transit finger↔table rate | 0.38 | **0.0** ✅ crit2 |
| transit min clearance (physical) | −0.008 | **0.0 (grazing, non-negative)** — crit3 needs > 0 |
| first strike vs first-over | ~50 / ~110 | strike ~232–310 or **never** (2/4) / over ~181–483 |
| lift / place | 0 / 0 | **0 / 0** |
| gate | FAIL | **FAIL (crit1+crit2 pass; crit3 grazing)** |

Two of four seeds **never touch the table in transit** (`first_fingtab = None`); the other two strike only *after*
they are over the object (allowed). The transit is clean.

## Why lift/place is still 0 — a grasp/descent problem (per your decision rule)

The route is **slow**: preshape ends ~step 96–230, so the arm reaches over-object late (first-over 181–483). Seeds
50000/50003 **do grasp** (~step 217) but then fail to lift within the horizon; seeds 50001/50002 don't grasp in
time. So lift/place = 0 is **not** a clearance failure — it is **grasp/descent pacing**: the slow monotonic route
eats the episode budget, starving the descend→grasp→lift→place tail. Your decision rule: *"if lift/place remain
zero despite positive clearance, diagnose grasp/descent separately, do not proceed to learning."*

## Answers to your checklist

- **Changed files:** `hymeko_rl/env/pick_place_env.py` only.
- **Gains changed?** **No** — kp60/kv15/dt÷2 frozen.
- **v1 behavior changed?** **No** — byte-identical; 15 tests green.
- **Route used:** `home → cf_mid_retract → HOLD → cf_hover` (monotonic, no down/up oscillation).
- **Commanded min clearance:** route waypoints are cf_mid ~+15 cm, cf_hover ~+6 cm; the raw validator reads +0.000
  = a **grip-transient artifact** (it samples the live grip qpos); the **clean physical transit confirms the
  commanded path is genuinely high**.
- **Physical min clearance:** **0.0 (transit, grazing, non-negative)** — up from −0.008 (the one negative reading,
  −0.012, is during the descent *over* the object, which is allowed).
- **Forbidden-pre-object rate:** **0.0** (from 1.0).
- **Transit finger/table contact rate:** **0.0** (from 0.38).
- **First-strike vs first-over-object:** ~232–310 (or never, 2/4) vs 181–483 — strike is after over-object or absent.
- **Lift / place:** 0.0 / 0.0.
- **Is gain tuning still justified?** **No** — the transit is now clean (physical clearance non-negative); the
  bottleneck moved to **grasp/descent (lift/place)**, not tracking. Gains are not the lever.
- **Is the 32-episode gate justified?** **No** — crit3 is grazing (0.0, not strictly positive) and lift/place = 0.

## Verdict / next

Clean transit achieved (crit1 + crit2). The remaining two items are **separate from clearance**:
1. **Grasp/descent (lift/place = 0):** diagnose the descend→grasp→lift→place tail — primarily the route's slowness
   starving the horizon (speed the route where the physical tracks / shorten the hold), and whether the grasp
   holds. Do **not** proceed to learning until this is understood.
2. **crit3 (grazing → strictly positive):** a small clearance-margin bump (the physical grazes 0.0 at one transit
   moment) — separate from the grasp issue.

Neither needs gains. Stopped after the v2 smoke; the 32-ep gate was not run. No BC/DAgger/RL.
