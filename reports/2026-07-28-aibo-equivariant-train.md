# The end-to-end equivariant trainer — post-hoc symmetrization WINS; in-loop equivariance suppresses discovery

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION.** Plan: `docs/plans/2026-07-28-aibo-symmetry-closure/`.
**Verdict: `POST_HOC_SYMMETRIZE_BEATS_IN_LOOP_EQUIVARIANCE` — impose symmetry AFTER discovery, not during.**

---

## The build

`MirrorEquivariantActor` wraps any squashed-Gaussian SAC actor so the policy is **exactly** mirror-
equivariant *during training*: the pre-squash mean is Reynolds-averaged over the order-2 mirror group
(`tanh` odd ⇒ the pre-squash mirror induces the post-squash action mirror). Duck-typed over the base's
internals (no change to the shared SAC module); adds no parameters; equivariance residual **0.0** before
and after gradient steps (5 unit tests). The mirror is the validated flat one, or the HyMeKo **structural**
automorphism (`structural_symmetry`), whose action permutation on `[fl,fr,bl,br]` is exactly the `[1,0,3,2]`
this uses.

## Result — in-loop equivariant training UNDERPERFORMS post-hoc symmetrization

Same 30k budget, omni crab, diagonal scaffold, `equivariance_residual = 0.0`:

| recipe | ‖action‖ | reach | +y | −y |
|---|---|---|---|---|
| raw unconstrained MLP | 0.82 | 0.40 | 1/2 | 0/2 (one-sided) |
| **train free → symmetrize post-hoc** (Phase B) | 0.71 | **0.60** | **1/2** | **1/2** (two-sided) |
| **train in-loop equivariant** (this) | **0.17** | **0.20** | **0/2** | **0/2** (symmetric-null) |

The in-loop equivariant policy converged to a **weak, symmetric, non-reaching** policy (‖action‖ 0.17 vs
0.71) — it reaches only straight, **neither** ±20°, symmetrically (+20° 0.407 ≈ −20° 0.509). Imposing the
symmetry **during** training **suppresses the symmetry-breaking exploration** that discovers the active
crab: SAC gets reward only if *both* sides work at once (a harder joint discovery), so it never breaks out
of the small-symmetric-action basin. The unconstrained actor *breaks* symmetry, finds the +y crab, and is
reinforced — after which post-hoc symmetrization mirrors it into a two-sided crab.

## Conclusion — the right end-to-end recipe

**Break symmetry to DISCOVER, then impose symmetry to GENERALISE.** The winning end-to-end trainer is
*train free + equivariant deploy* (Phase B: reach 0.60, two-sided) — **not** a hard-equivariant training
constraint (this: reach 0.20, symmetric-null). The HyMeKo structural automorphism is the **correct group
action** (it *is* the mirror Phase B used), but it belongs as a **post-hoc projection at deploy**, not as a
training-time constraint. This is a clean, slightly counter-intuitive lesson: equivariance is a
generalisation prior, and applying it before the behaviour exists prevents the behaviour from forming.

## Warm-start probe — also degrades (with a confound)

Warm-starting the equivariant actor from the raw active-crab policy (discover-then-constrain) *also*
collapsed to reach 0.20 / symmetric-null. **Confound:** the critic was fresh (not warm-started), so the
untrained critic misled the good actor before it converged — so this does not cleanly isolate the
constraint. But combined with the from-scratch result, the practical conclusion is unchanged and
strengthened: **post-hoc symmetrization at deploy is the robust winner (0.60, two-sided); every in-loop
variant tried degrades to symmetric-null (0.20).** A clean warm-start (actor + critic together) is the
open follow-up if in-loop equivariance is to be salvaged.

## Bearing on the HSiKAN thread

The trained HSiKAN had a **null residual** (rode the scaffold) — it never discovered an active crab, so
neither post-hoc symmetrization (nothing to mirror) nor in-loop equivariance (suppresses discovery further)
yields a two-sided HSiKAN crab. Both routes need the policy to **first learn an active crab**; that is an
exploration/authority problem upstream of the symmetry, not solved by equivariance. The natural next probe
is a **warm-start**: initialise the equivariant actor from the raw active-crab policy and continue — keeping
the discovered crab while making it two-sided (best-of-both). Recorded, not yet run.

## Files / tests

```
scenarios/aibo/equivariant_actor.py            NEW  MirrorEquivariantActor (in-loop Reynolds symmetrization) + flat mirror + equivariance_residual
scenarios/aibo/run_aibo_equivariant_train.py   NEW  trains the in-loop equivariant MLP, evals both sides
tests/test_aibo_equivariant_actor.py           NEW  5 tests: involutions, exact equivariance, no added params, equivariance survives a grad step
reports/2026-07-28-aibo-residual-trot/result_equivariant_mlp_diag.json  NEW
```

`ruff` clean; **120/120** AIBO tests green. CPU, seed 0, ~380 steps/s, peak RSS < 0.5 GB. CORE.YAML: none.
Scope: 1 seed; the qualitative ordering (post-hoc two-sided 0.60 ≫ in-loop symmetric-null 0.20) is large and
mechanistically explained (exploration vs constraint).
