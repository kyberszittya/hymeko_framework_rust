# v2 Cartesian / clearance-controlled preshape (2026-07-07)

## Summary

Replaced the joint-space preshape with a **Cartesian / clearance-controlled** one: progress the commanded config
toward `cf_hover`, then **enforce a validated commanded finger-table clearance floor** each step by lifting the
tool (xy held) until clearance ≥ `_V2_CLEAR_FLOOR` (0.04 m). **Gains stay frozen** (kp60/kv15/dt÷2); v1 byte-identical.
**It achieves its stated goal — the commanded trajectory is raised off the table (preshape commanded clearance
+0.069–0.075 m, ~7 cm, up from +0.000)** — but **physical clearance stays negative (−0.008 m)**. Per your decision
rule: **commanded positive, physical negative → residual tracking gap; do not tune gains.**

## What changed (command-path geometry only; gains untouched)

- `_v2_preshape_target` → a single multi-start `cf_hover`; `_v2_preshape_step` now progresses toward it and, each
  step, **validates the commanded finger-table clearance** (`_v2_cmd_clearance`, a scratch-`MjData` FK + `mj_geomDistance`)
  and **lifts the tool (xy held) until clearance ≥ floor** — a command-side clearance floor.
- Preshape rate slowed 0.22→0.05 so the physical arm tracks the high commanded path instead of lagging behind a
  fast-jumped target.
- The physical `pick_clearance` harness remains the **sole safety authority**; this is a controller-side validation.

## Results (v2 smoke, 4 ep, seeds 50000–50003)

| metric | v2 (gain-fix, joint preshape) | **v2 Cartesian preshape** |
|---|---|---|
| **commanded** min clearance (preshape) | +0.000 (grazes) | **+0.069 … +0.075 m (~7 cm)** ✅ goal met |
| physical min clearance | −0.005 | **−0.008 m** |
| forbidden-pre-object rate | 0.5 | 1.0 |
| transit finger↔table rate | 0.047 | 0.38 |
| first strike vs first-over | ~127 / ~180 | ~50–72 / ~106–171 |
| lift / place | 0 / 0 | 0 / 0 |
| gate | FAIL | FAIL |

**The commanded path is genuinely raised** (measured +7 cm during preshape, all four seeds; the +0.000 readings in
the step-trace are single-step measurement artifacts — the *same* arm config reads +0.064 on both neighbours — plus
the descend phase which commands grasp_z over the object, allowed). **But the physical arm still lags on its servo
path to the target and dips to −0.008**, and — with a single-target route — the physical result did *not* improve
over the joint two-stage (it regressed: forbidden 0.5→1.0). Command-side shaping raised the command; it did not
close the physical tracking gap.

## Decision-rule answers (your checklist)

- **Changed files:** `hymeko_rl/env/pick_place_env.py` only.
- **Gains changed?** **No** — kp60/kv15/dt÷2 frozen exactly.
- **v1 behavior changed?** **No** — v1 re-smoke byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577); 15 tests green.
- **Commanded min clearance:** **+0.069–0.075 m** during preshape (comfortably positive, several cm).
- **Physical min clearance:** **−0.008 m min / −0.007 mean**.
- **Forbidden-pre-object rate:** 1.0.
- **Transit finger/table contact rate:** 0.38.
- **First-strike vs first-over-object:** ~50–72 vs ~106–171.
- **Lift / place:** 0.0 / 0.0.
- **Is the 32-episode gate justified?** **No** — physical clearance still negative.

## Verdict (your decision rule)

> "If commanded clearance is positive but physical clearance remains negative, then report the residual tracking
> gap, but do not tune gains automatically."

This is exactly the case: **commanded clearance is comfortably positive (+7 cm); physical clearance remains negative
(−0.008 m).** The residual is the **physical tracking gap** — the arm lags the high commanded path by ~8 cm during
the dynamic catch-up (steady-state sag ~2 cm after the frozen gain fix; the dynamic transient is larger). Command
geometry is no longer the bottleneck; **tracking is.** Gains are frozen per your instruction, so I stop here and
report rather than tune.

**Note (honest):** the single-target Cartesian route physically *regressed* vs the joint two-stage (forbidden
0.5→1.0) because the servo's path to one far target dips more than a route guided through the retracted `cf_mid`
intermediate. If we want the physical arm guided high without touching gains, a **monotonic multi-waypoint
high-z Cartesian route** (cf_mid → cf_hover, each clearance-floored, non-oscillating) is the logic-only option; the
alternative is un-freezing the gain/tracking lever. Both are your call.

## Provenance

ruff clean · 15 tests green · v1 byte-identical. Stopped after the v2 smoke; 32-ep gate not run. No BC/DAgger/RL.
