# The stepping frontier — thoroughly ruled out in this balance env (honest negative)

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` (head at start `12edd7ff`) · direction **1** (stepping action space).

## Question
A/2 identified **stepping** (capture-point foot placement) as the capability the in-place scaffold lacks. Does
giving the humanoid stepping authority extend recovery beyond the certified in-place viability boundary?

## What I tried (all Δ = +0.00 vs the scaffold)
| approach | perturbation | result |
|---|---|---|
| arm reaction-wheel residual | base pitch-rate | Δ 0 |
| whole-body learned linear residual (16 joints, CEM) | base pitch-rate | Δ 0 (couldn't even overfit train) |
| **wide-action** learned residual (`delta_scale 0.4→1.5`, stepping authority) | base pitch-rate | Δ 0 |
| **structured** scripted protective step (hips/knees/ankles, timing swept) | base pitch-rate | Δ 0 |
| reactive lateral capture-point step (hip abduction) | **lateral push** | Δ 0 |

## Diagnosis — three honest reasons stepping can't be shown here
1. **Wrong perturbation for stepping (angular).** The main perturbation is a base **angular** velocity
   (`qvel[4]`, up to ~4 rad/s) — the body *rotates*. Capture-point stepping recovers a *translating* CoM (a push),
   not a rotation; you cannot step to catch a spin. So the pitch-rate failures are structurally out of stepping's
   reach.
2. **Sharp cliff, not a gradual band (lateral push).** For the *right* (linear) perturbation, the scaffold is a
   **hard cliff**: it recovers lateral pushes up to ~2.3 m/s and then fails **completely** (survive 1.00 at 2.2 →
   0.00 at 2.5). There is no gradual "stepping-recoverable" band above the in-place limit — the push at the cliff
   is instantaneous and violent, and the fall gate (`pelvis_z < 0.55`) fires before any ~0.3 s step could finish.
3. **The env is not designed for stepping.** Reset plants both feet; the action is a bounded PD-hold-`q0` offset;
   a real protective step must lift a foot (break contact) and execute a phased weight-shift → swing → plant — a
   **structured trajectory**, not the reactive/linear feedback these controllers are. (Widening the action to
   1.5 rad and hand-scripting a step still did nothing.)

## What is captured (positive characterisation)
The certified scaffold's **viability region** is now bounded in two dimensions:
- pitch-rate: **~3.25 rad/s** (bisection, `recoverable_bound`);
- lateral push: **~2.3 m/s** (bisection, `lateral_push_bound`) — a sharp cliff.
Both are exposed + tested; the HSTL monitor (from C) certifies/flags within them.

## Honest conclusion
**Stepping cannot be demonstrated as beneficial in this env** — not because stepping is worthless, but because
(a) the dominant perturbation is angular (stepping-irrelevant), (b) the linear-push failure is an abrupt cliff,
and (c) the env + action space are not built for a phased step. Demonstrating stepping needs a **new setup**: an
impulsive-push perturbation model, a stepping-capable env (foot-lift allowed, longer horizon, contact scheduling),
and a **structured capture-point planner** (a trajectory / FSM), not reactive feedback — a separate build, flagged.

## Files
`scenarios/humanoid/humanoid_certified_balance.py` (+22: `lateral_push_bound` / `lateral_push_recovers`),
`tests/test_humanoid_certified_balance.py` (+1 test), this report. No dependency change, no §1.

## Tests
`pytest tests/test_humanoid_certified_balance.py` → 4 passed (incl. the lateral-push cliff); ruff clean.

## Provenance
Git SHA at start `12edd7ff`. Env: `.venv` (Python 3.11, MuJoCo, NumPy 2) + built CLI, macOS. Seeded.
