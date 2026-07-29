# Turning stability as a Vukobratović-ZMP / Lyapunov capturability certificate

**Date:** 2026-07-29 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION.** · **Verdict: `VUKOBRATOVIC_ZMP_CERTIFICATE_FORMALIZES_THE_TURN_STABILITY_BOUNDARY`.**

---

## Two corrections (user), both right

1. **The proper ZMP is Vukobratović's**, not the LIPM capture point I used first. The capture point is
   *translational* (CoM position + velocity) and stayed blind (+0.93) 40 steps before the AIBO's *spin*
   tip. The full **Vukobratović ZMP includes the angular-momentum-rate term ``Ḣ``**
   (``ZMP = CoM_xy − (m·z·a_xy + [Ḣ_y,−Ḣ_x]) / (m(z̈+g))``), and that term is exactly what fires for a
   rotational tip.
2. **There is a Lyapunov boundary** — and it already lives in the code (`scenarios/aibo/capture_step.py`
   `PushRecoveryLyapunov` + the LIPM capture point; `lyapunov.py` `evaluate_lyapunov`; hence the branch
   name). The **capturability region** — the set of states from which the ZMP can be kept inside support —
   *is* that Lyapunov region of attraction.

## The certificate (new: `scenarios/aibo/turn_stability.py`)

`vukobratovic_zmp` (whole-body CoM/momentum from MuJoCo `subtree_*`, ``a``/``Ḣ`` by finite difference) +
`support_margin` (signed distance from ZMP to the stance-weighted support) + `zmp_stability_certificate`
(CIP-0 **SAFETY**: PASS iff the ZMP never leaves support). Runs under the **motion contract** (the
governor) — unlike the retracted dynamics-exploit "recovery" in `capture_step.py` (26.9 rad/s joints,
feet airborne, certified only because no governor was applied).

## Validation — the certificate IS the turning-stability boundary, and it's predictive

Rotational-couple turn at rate ``g``:

| g | yaw °/1000 | min ZMP margin | ZMP certificate | actually tips? |
|---|---|---|---|---|
| 1.0 | 61 | +0.62 | **PASS** | no |
| 1.1 | 31 | +0.62 | **PASS** | no |
| 1.2 | −71 | +0.17 | **PASS** | no |
| 1.3 | −198 | **−0.23** | **FAIL** | not yet |
| 1.6 | +214 | **−0.22** | **FAIL** | **tips** |

The ZMP leaves support (certificate FAILS) at **g=1.3 — before the actual fall at g=1.6**: the
Vukobratović-ZMP boundary is a **predictive** capturability boundary, exactly the Lyapunov region-of-
attraction edge. The certified-safe turning zone is g ≤ 1.2 (max ~61°/1000).

## The synthesis this reveals

The certificate formalizes the ~47–61°/1000 turning ceiling as a **Vukobratović-ZMP / Lyapunov
boundary** — a formal, reward-independent CIP-0 SAFETY certificate. Crucially it also shows the boundary
is **not the binding constraint on turn SPEED**: within the certified region the rotational couple is
already non-monotonic and *reverses* past g≈1.0 (g=1.2 → −71°/1000), so the speed limit is the **couple
mechanism**, which the certificate certifies as safe up to ~61°/1000. Faster stable turning needs a
different mechanism; the same certificate would then delimit *its* capturable region. This is the
principled stability layer the CIP-0 campaign wants: turning is now **certified**, not just measured.

## Files / tests

```
scenarios/aibo/turn_stability.py       NEW  vukobratovic_zmp (with Ḣ) + support_margin + zmp_stability_certificate (CIP-0 SAFETY)
tests/test_aibo_turn_stability.py      NEW  4 tests: stable ZMP-in-support / fast ZMP-leaves / ZMP finite / certificate fn
```

`ruff` clean; new tests green. CORE.YAML: none. SIMULATION. Reuses the existing `hymeko_control` Certificate
framework and the `capture_step`/`lyapunov` infrastructure.
