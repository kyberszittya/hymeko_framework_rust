# AIBO goal-reaching SOLVED — deterministic turn reaches 100%; the ±135° "wall" was HORIZON, not tipping

**Date:** 2026-07-29 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Diagnostic.** · **Verdict: `DETERMINISTIC_TURN_THEN_WALK_REACHES_100PCT_GIVEN_HORIZON`.**

---

## The finding

Chasing the ±135° failure, I measured the balance (foot-support) entropy — it cleanly flags tipping
(H≈1.0 stable, →0 tipping) — and tested whether a *balance-gated* faster turn could reach the wide
bearings. It couldn't (reactive gating tips as fast as the ungated fast turn). But the decisive check was
simpler: **is ±135° tipping-limited or horizon-limited?** The stable g=1.0 turn was *already* rotating
toward 135° (0.6→0.401 at horizon 2400), just cut off. Giving it more time:

| horizon | wide bearings (±90/±135, 8 goals) | full 18-goal grid |
|---|---|---|
| 2400 | 0/8 | 10/18 = 0.56 |
| 4000 | 6/8 | 15/18 = 0.83 |
| **6000** | **8/8** | **18/18 = 1.00** |

**The deterministic rotational-couple `turn_then_walk` reaches every goal (100%)** — 0°, ±20/40/90/135°,
both distances — and **never tips** (upright ≈ 0.99 throughout). The ±135° "wall" was the **2400-step
horizon** cutting off a stable-but-slow 135° turn mid-rotation, not a stability limit.

## What this means (honestly)

1. **The main goal — AIBO reaches a designated target at any bearing — is SOLVED, deterministically, at
   100%.** The lever was the *mechanism* (the rotational-couple turn, 0.11→0.50→1.00), plus an adequate
   time budget for wide-bearing turns.
2. **There is no tipping wall to overcome.** The stable turn is always upright; you only need fast turning
   if you insist on a short horizon. So the **balance-entropy / fast-turn direction targets a non-problem
   for this goal** — the stable turn already reaches everything given time.
3. **The RL (residual) and HSiKAN-vs-MLP comparisons were run at horizon 2400**, which artificially capped
   the wide bearings — so they were optimising against a horizon artifact, not a real limit. With an
   adequate horizon the deterministic scaffold is at 100%, leaving nothing for a residual to add. This is
   consistent with those runs finding the residual ≤ scaffold and HSiKAN ≈ MLP.

## Where the balance entropy would still matter (a different objective)

The foot-support balance entropy is a **real, validated** signal (H≈1.0 upright, →0 tipping). It is not
needed for goal *reachability* (the stable turn suffices), but it is exactly the signal for **time-
efficient** turning: turning *fast* (toward the ~169°/1000 regime) to reach wide bearings in fewer steps
*without tipping* is a genuine rate-vs-stability control problem, and a predictive (learned) policy using
the balance entropy — where a reactive gate failed — is the honest way to attack it. That is a **speed**
objective, distinct from the **reachability** goal, which is now solved.

## Files

```
reports/2026-07-29-aibo-goal-reaching-solved.md   NEW  (diagnostic; no code change — uses the existing turn_then_walk)
```

CPU, seed-varied goals, MuJoCo. CORE.YAML: none. SIMULATION.
