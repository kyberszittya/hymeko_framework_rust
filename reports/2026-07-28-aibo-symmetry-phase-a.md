# Phase A — the symmetric scaffold: crab-asymmetry diagnosis CONFIRMED (with an honest substrate limit)

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION.** Plan: `docs/plans/2026-07-28-aibo-symmetry-closure/` (4 artifacts, on disk).
**Verdict: `SCAFFOLD_ASYMMETRY_CONFIRMED_BOUND_RESTORES_MIRROR_BUT_CANNOT_LOCOMOTE`.**

---

## The test

The omni crab is one-sided (+y 2/2, −y 0/2) across 8 policy variants. The diagnostic concluded it is a
**scaffold-induced dynamics asymmetry**: the abduction residual sits over a **diagonal** trot
`(0,π,π,0)` whose two body sides are in different gait phases at every instant. Phase A adds a
configurable gait phase and tests the **bound** pattern `(0,0,π,π)` — instantaneously left-right
symmetric (fl==fr, bl==br) — asking: does a symmetric substrate restore the crab's mirror symmetry?

## Result 1 — mechanism CONFIRMED (the decisive, cheap test)

Torso lateral displacement `dy` from a **constant** abduction pattern over each scaffold, stable regime
(`abd_scale=0.5`, upright throughout):

| pattern | **diag** (asymmetric trot) | **bound** (symmetric) |
|---|---|---|
| left `[1,0,1,0]` | dy −0.119 (up 0.98) | dy **+0.094** (up 0.90) |
| right `[0,1,0,1]` | dy −0.094 (up 0.80) | dy **−0.078** (up 0.82) |
| **sign product** | **> 0 (same side)** | **< 0 (mirror-opposite)** |

Over the diagonal trot, left- and right-side abduction push the **same** lateral way (no mirror
symmetry for a policy to exploit → one-sided crab). Over the **bound** scaffold they push **opposite**
ways — the **left-right mirror antisymmetry is restored**. This is the direct confirmation that the
one-sidedness is **scaffold-induced**, not a policy/representation gap. Pinned by
`test_bound_restores_mirror_antisymmetry_of_abduction` (a measured regression test).

## Result 2 — behavioral: symmetric, but bound cannot reach the grid

A per-node HSiKAN omni trained over the bound scaffold (30k, same fast config) on the ±{0,20,40}°,
0.6 m grid: **reach 0.0, +y 0/2, −y 0/2** — but the per-goal min-distances are **near-identical across
sides** (+20° 0.590 / −20° 0.586; +40° 0.580 / −40° 0.600). So bound fails **symmetrically** (both
sides equal), unlike diag's one-sided +y 2/2. The symmetry is restored *behaviorally* too — the crab is
just too weak to reach the far grid.

**Why:** the `a=0` bound scaffold is **stable but stationary** — upright 0.996, yet min_dist stays at
0.60 (it steps in place, no net forward motion). The bounded abduction (`abd_scale=0.5`) crabs only
~0.1 m laterally over a full rollout — a genuine, *symmetric* two-sided crab (left→+y, right→−y,
upright), but small-amplitude. The 0.6 m grid is out of range.

## Result 3 — the honest limit: pushing harder tips it

Doubling the abduction authority (`abd_scale=1.0`) to crab further **tips the robot** (upright −0.5,
both sides) and **destroys the antisymmetry** (both sides then push +y). So the clean mirror antisymmetry
holds **only in the stable small-signal regime**; the symmetric substrate trades away locomotion power,
and cannot be pushed into a strong reaching crab without falling.

## Conclusion

The core symmetry question is **answered**: the omni-crab one-sidedness is a **scaffold-induced
dynamics asymmetry**, confirmed two ways — (1) the bound scaffold restores the mirror antisymmetry of
abduction (mechanism), and (2) a policy trained over bound fails **symmetrically** rather than one-sided
(behavior). This reinforces the recurring campaign theme: **the AIBO scaffold is the binding
constraint**. The honest limit is that instantaneous left-right symmetry and forward locomotion are in
tension for a clock gait — the symmetric (bound) substrate is a weak locomotor, so a **strong two-sided
reaching crab** on the 0.6 m grid is not achievable over bound at this task scale. Demonstrating one
would need either a **closer goal grid** (where the small symmetric crab reaches both sides) or a
genuinely symmetric-**and**-locomoting substrate (a co-designed lateral gait) — a scaffold redesign, not
a policy change.

## Files / tests / provenance

```
scenarios/aibo/locomotion_gait.py       MOD  GAIT_PHASES {diag,bound,pace,pronk} + SteeredTrotGait.phase field
scenarios/aibo/residual_trot.py         MOD  ResidualTrotConfig.gait_phase, threaded through omni apply + obs
scenarios/aibo/run_aibo_hsikan_omni.py  MOD  --gait
tests/test_aibo_gait_phase.py           NEW  5 tests (patterns / bound instantaneous symmetry / diag backward-compat / invalid raises / MEASURED bound-restores-antisymmetry)
reports/2026-07-28-aibo-residual-trot/result_hsikan_omni_signedkan_per_node_bound.json  NEW
```

`ruff` clean; **107/107** AIBO tests green. CPU, seed 0, MuJoCo; peak RSS ~0.38 GB (« 16 GB cap).
CORE.YAML touched: none. `gait_phase="diag"` reproduces the prior scaffold (backward-compatible default,
regression-pinned).
