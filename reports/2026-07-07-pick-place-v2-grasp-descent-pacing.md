# v2 grasp/descent pacing diagnosis + minimal fix (2026-07-07)

## Summary

Diagnosed the per-phase timing, then applied a **minimal, clearance-safe pacing fix** (faster preshape: rate
0.05→0.10, cf_mid hold 20→8, Cartesian caps 0.03/0.02→0.05/0.03). **Gains frozen; v1 byte-identical.** The preshape
now ends ~step 47 (was ~97) and **clean transit is preserved** (crit1/crit2 pass). But **lift/place stay 0**, and
the diagnosis shows *why the pacing fix alone can't finish it*: the horizon sink is **not** the preshape — it's the
**physical arm's slow convergence over the object** (gain-limited) for far spawns, and a **grasp that is lost after
commit** for near spawns. Neither is command pacing.

## Phase timing (v2, seeds 50000–50003, after the pacing fix)

| seed | over-obj | descend | close | contact | commit(grasp) | lift-cmd | grasped | lifted | rem-after-grasp | final phase |
|---|---|---|---|---|---|---|---|---|---|---|
| 50000 | 168 | 168 | 170 | 200 | 211 | 211 | **yes** | no | **409** | HOLD_HOVER |
| 50001 | 334 | 334 | 335 | — | — | — | no | no | 0 | CLOSE |
| 50002 | 489 | 489 | — | — | — | — | no | no | 0 | DESCEND |
| 50003 | 171 | 171 | 173 | 201 | 212 | 212 | **yes** | no | **408** | GRASP_DWELL |

Phase entries (seed 50000, representative): `cf_mid@0 · hold@33 · cf_mid→cf_hover@40 · HOLD_HOVER@47 · DESCEND@168 ·
CLOSE@170 · GRASP_DWELL@200 · CARRY@211`. The preshape is ~47 steps; **`HOLD_HOVER` runs 47→168–489** — that is the
sink, and it is the *physical arm crawling to within 6 cm of the object*, not the command.

## Two distinct blockers (neither is command pacing)

1. **Near objects (50000/50003): grasp mechanics.** They reach over-object early (168/171), grasp at ~211 with
   **~409 steps still free**, then **lose the grasp after commit** (CARRY → back to HOLD_HOVER/GRASP_DWELL, never
   lifting). Budget is *not* the problem — the grasp does not hold through the lift.
2. **Far objects (50001/50002): physical convergence.** `HOLD_HOVER` waits for the physical tool to reach
   `horiz ≤ 0.06`; for the far spawns that takes until step 334/489 (near the reach edge, the sag-limited servo
   converges the last few cm very slowly). **Speeding the *command* did not move this** (over-object 181→168 for
   near, 483→489 for far) — it is the physical/gain-limited convergence, frozen.

## Results (v2 pacing smoke, 4 ep)

- forbidden-pre-object rate: **0.0** (crit1 PASS, preserved) · transit finger↔table rate: **0.0** (crit2 PASS)
- physical transit min clearance: **0.0** (grazing, non-negative; crit3 needs > 0)
- first-strike vs first-over: strike ~207–212 or **never** (2/4) vs over 168–489 — strike after over-object or absent
- **grasp rate 2/4 · lift rate 0/4 · place rate 0/4**
- final-phase distribution: HOLD_HOVER, CLOSE, DESCEND, GRASP_DWELL (one each)
- gate: **FAIL** (crit1+crit2 pass; crit3 grazing; lift/place 0)

## Checklist answers

- **Changed files:** `hymeko_rl/env/pick_place_env.py` only (3 pacing constants).
- **Gains changed?** **No** (kp60/kv15/dt÷2 frozen).
- **v1 behavior changed?** **No** — byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577); 15 tests green.
- **Clean-transit guarantees:** preserved — forbidden-pre-object 0.0, transit contact 0.0, physical clearance
  non-negative, no new oscillation.
- **Grasp / lift / place rate:** 2/4 · 0/4 · 0/4.

## Decision — next step is GRASP TUNING (not a clearance bump, not the gate)

Per your decision rule:
- *"clean transit preserved"* → **yes**, keep the pacing change (it is safe and frees preshape budget).
- *"lift/place remain zero despite enough remaining horizon after grasp → diagnose grasp mechanics separately"* →
  seeds 50000/50003 (grasp at ~211, **~409 steps free**, grasp lost) → **grasp mechanics is the clear next step.**
- *"lift/place remain zero because the route still consumes the horizon → further pacing justified"* → seeds
  50001/50002 consume the horizon, **but the consumption is physical convergence to the far object, not command
  pacing** — so *command* pacing is exhausted; this residual is gain/reach-limited (frozen) and is a separate,
  out-of-scope lever.

**So: the next isolated step is grasp mechanics tuning** (why the grasp is lost through the descent→commit→lift
transition — dwell length, lift onset, grip firmness at the v2 grasp pose), **not** a clearance-margin bump and
**not** the 32-episode gate. The far-object physical-convergence residual is flagged as a separate gain/reach item
(gains frozen per your instruction).

Stopped after the smoke; 32-ep gate not run. No BC/DAgger/RL.
