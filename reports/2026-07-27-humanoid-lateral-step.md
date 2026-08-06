# Humanoid protective step — lateral residual over the certified scaffold (honest negative)

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION. Genuine RL.** · **Verdict: `LATERAL_RESIDUAL_DOES_NOT_EXTEND_CERTIFICATION` (0/12 both; cross-plane dissociation vs the sagittal positive).**

---

## Goal (user option a)

Now that the Vukobratović frontal DOF exist, learn a **protective step** for strong lateral
pushes — via a bounded residual over the certified PD-hold scaffold (coin-R8), on the
regime where the scaffold degrades.

## Feasibility probes (measured first, §3)

- **The 16-DOF frontal PD is a strong lateral balancer.** With the added abduction + ankle
  roll held at `q0`, the fixed-stance PD survives lateral pushes to **~1.5 m/s** (500/500 at
  1.0 m/s), degrading only beyond ~2.0. (A first probe that showed failure at 0.6 m/s was a
  **probe bug** — it didn't reset `self._t`, so every rollout truncated after 1 step; the env
  reset is correct.)
- **The foot barely lifts.** A scripted weight-shift + lift clears **0.00 m at delta_scale
  0.4** and only **+0.055 m at 1.8** (a foot is 5 cm thick). Naive scripted step/widen
  controllers **underperform** the fixed stance at strong pushes (2.5 m/s: fixed 386 vs step
  373 vs widen 343 upright steps) — they destabilize more than they help.
- **Well-posed regime:** at lateral push **1.8–3.0 m/s** the fixed-stance **survives but
  certifies 0/12** (the COM drifts laterally, V never converges) — the same survival≠stability
  gap the sagittal residual exploited.

## Result — the residual does NOT extend lateral certification

Residual SAC over the scaffold (push 1.8–3.0, delta_scale 1.0, ANNEAL + best-val ckpt,
150 k steps). **Validation certified rate = 0 on all 15 evals.** Held-out test (12 seeds):

| policy | certified | mean survival | V_final | V_max | foot-lift | mean \|a\| |
|---|---|---|---|---|---|---|
| PD-hold scaffold (a=0) | **0 / 12** | 1.00 | 2.17 | 2.18 | 0.13 m* | 0.00 |
| SAC residual (best-val) | **0 / 12** | 1.00 | 2.36 | 2.44 | 0.14 m* | 0.39 |
| delta | 0 | −0.00 | **+0.19 (worse)** | — | +0.01 | — |

`RESIDUAL_MATCHES_SCAFFOLD` (delta 0). *The 0.13 m "foot-lift" is **passive** — a strong
lateral push tips the body and a foot comes off the ground; it is not a controlled step
(the scaffold's \|a\| is 0). The residual **acts** (\|a\| 0.39) but does **not** reduce V,
improve survival, or produce a real step — Vfinal is slightly worse.

## Honest interpretation — cross-plane dissociation

The certificate requires the COM to return **over the support, at rest** (V ≤ 0.05). Under a
strong lateral push both policies keep the pelvis nominally upright (lenient fall threshold)
but the **COM drifts laterally and stays there** (V ~ 2.2) — never recentered. To recenter
after a strong lateral push you must either (a) reverse the lateral momentum with frontal
ankle/hip authority (insufficient here), or (b) **take a protective step** — place a foot
under the drifted COM. The step is **marginal on this model** (foot clears ≤ 5.5 cm only at a
4.5× action range), and a residual over the position-servo scaffold does not discover one.

This contrasts sharply with the **sagittal** result (`…-humanoid-sac-residual.md`): there the
residual extended the certified envelope **0 → 12/12**, because sagittal recovery-to-rest is
achievable with in-plane ankle/hip. **Sagittal recentering = learnable in-plane; lateral
recentering = needs a step = not learned here.** An honest, informative negative.

## Files touched

```
scenarios/humanoid/balance_env.py         +push_lat_lo/hi (lateral push) + w_step (capture-point step shaping) + _capture_step()
scenarios/humanoid/run_humanoid_sac.py    +--lateral mode (1-step-reachable band, delta_scale, w_step); best-ckpt guard
reports/2026-07-27-humanoid-lateral-step/{sac_residual_gates.json, humanoid_sac_residual_best.pt, sac_train_*.log}
```

## Tests / lint

`ruff` clean. Balance-env tests pass (8/8) with the new `push_lat` field. The certificate was
**not** modified. Peak RSS ≈ 0.3 GB, wall ≈ 10 min.

## Levers (2) foot-clearance + (3) step-shaping reward — tried, still negative

Per the follow-up, I added **(2)** a `w_step` **capture-point step-shaping** reward
(`xi_y = com_y + com_y_vel·√(com_z/g)`; reward a foot near the capture point — the
Pratt/Koolen capturability criterion, the Vukobratović-lineage step condition) and **(3)**
**higher foot-clearance actuation** (delta_scale up to 1.8). The Lyapunov certificate stayed
the reward-independent gate. Three residual runs, all **0/12 certified**:

| run | push | delta_scale · w_step | certified | foot-lift (vs fixed) | capture_err (vs fixed) | Vfinal (vs fixed) |
|---|---|---|---|---|---|---|
| v1 | 1.8–3.0 | 1.0 · 0 | 0/12 | +0.01 m | — | +0.19 (worse) |
| v2 | 1.8–3.0 | 1.8 · 1.0 | 0/12 | **+0.13 m** (step gesture) | +0.02 (no better) | −0.23 |
| v3 | **0.8–1.5** (1-step-reachable) | 1.0 · 0.5 | 0/12 | +0.07 m | +0.02 (no better) | **+0.50 (worse)** |

Measured boundary: **fixed-stance certifies lateral pushes only to ~0.5 m/s** (0.75 at 0.5,
0 at ≥0.7), so v3 targeted the band where the scaffold fails but the capture point is within
reach. Findings:

- **A step *gesture* emerges** with clearance+shaping (v2 foot-lift 0.256 m vs 0.130 m
  passive) — the levers do induce more foot motion.
- **But never a *functional* capture-point step:** `capture_err` is **not reduced** in any
  run (the lifted foot is not placed to catch the COM), and certification stays 0/12.
- **The residual often makes it slightly worse** (v1/v3 Vfinal higher) — its step gestures
  disrupt the strong frontal-PD near-recovery more than they help.

## Bottom line (robust, 3 honest attempts)

On the Vukobratović 16-DOF humanoid the frontal-plane fixed-stance PD is a **strong lateral
balancer**, and a learned residual — even with **capture-point step-shaping and higher foot
clearance**, and even in the **1-step-reachable band** — **does not** extend Lyapunov
certification to lateral pushes (0/12 across three configs; often slightly worse). A step
*gesture* is inducible, but a *functional* protective step (place the swing foot at the
capture point, recenter the COM to rest) is not discovered by residual RL on this model.

This is a robust **cross-plane dissociation**: the **sagittal** residual extended
certification **0 → 12/12** (in-plane ankle/hip recentering is a learnable bounded residual);
the **lateral** residual does not (recentering needs a step). Reported honestly as a negative
across the tried levers; the sagittal positive stands. A genuine protective step would need a
model-based capture-point stepping *scaffold* (not a residual over the fixed stance) or a
richer swing-phase actuation — future work, not claimed.
