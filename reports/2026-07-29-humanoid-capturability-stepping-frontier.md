# The humanoid stepping frontier, framed — Pratt/Koolen N-step capturability over the ZMP core

**Date:** 2026-07-29 (JST) · **Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION.** · **Verdict: `STEPPING_FAILURE_IS_THE_1_STEP_CAPTURABLE_REGION` (framed, not solved).**

---

## Why

Protective STEPPING is the humanoid's open negative (0/12 certified — the lateral step doesn't catch the
COM). Part (1) gave a genuine ZMP-in-support certificate; part (2) frames *when a step is required and
where it must land*, with **Pratt/Koolen N-step capturability** over the shared stability core — turning
"stepping fails" into a formal boundary.

## What was added (`hymeko_control.stability`, shared core)

- `capture_point` — the LIPM capture point ``ξ = CoM_xy + CoM_vel·√(z/g)`` (where the CoM comes to rest;
  the *translational* capturability measure — the right one for a fall-over, vs the rotational ZMP).
- `capturability_level` — Pratt/Koolen N-step class of ``ξ``: **0** = 0-step (``ξ`` in support: balance in
  place), **1** = 1-step (``ξ`` in support ⊕ one step of reach: MUST step), **2** = not capturable (fall);
  returns the signed 0-step and 1-step margins.

## Result — the 0/12 failure IS the 1-step region

Humanoid PD scaffold under increasing lateral push (peak over the episode):

| lateral push | capturability | 0-step margin | meaning |
|---:|:---:|---|---|
| 0.0–1.0 | **0** | +0.09 → +0.01 | balance in place — the certified PMP/IDA-PBC basin |
| 1.5 | **1** | −0.08 | **MUST step** (ξ left support, within 1-step reach) |
| 2.0 | **1** | −0.16 | **MUST step** |

The push that the in-place controllers survive (≤ 1.0) is exactly the **0-step** region; the push that
needs a step (≥ 1.5) is the **1-step** region — precisely where protective stepping is 0/12. So the
failure is **not** that the humanoid enters an uncapturable state; it enters a **1-step-capturable** state
(a step *could* save it) and the controller fails to place the foot so the new support contains ``ξ``.

## What this gives (honest)

- A **formal target** for the step: the step must move the support polygon to contain the capture point
  ``ξ`` (make ``capturability_level`` return to 0 after the step). The 0/12 residuals never did this
  ("the lifted foot is not placed to catch the COM"); the certificate says *exactly* where it must land.
- A **decision boundary**: 0-step → hold (PMP/IDA-PBC, certified); 1-step → step to capture ``ξ``;
  uncapturable → unavoidable fall. The controller can *switch* on the capturability level.

This **frames** the open stepping problem as a capturability-targeted control problem — it does **not**
claim to solve it (stepping stays 0/12). The next attempt is a step whose foot placement is *driven by*
the capture point (place the swing foot at ``ξ`` + a stability margin), gated by the certificate.

## Files / tests

```
hymeko_control/stability.py                  MOD  + capture_point + capturability_level (shared, synced to AIBO)
tests/test_humanoid_zmp_stability.py         MOD  +1 test (capture point + 0/1/2-step classification)
```

`ruff` clean; 6/6 humanoid ZMP tests + full humanoid suite green; AIBO turn-stability green (core synced,
byte-identical). CORE.YAML: none. SIMULATION.
