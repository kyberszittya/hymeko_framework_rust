# Phase B — hard mirror-equivariance FIXES the one-sided crab (my hypothesis was falsified)

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION.** Plan: `docs/plans/2026-07-28-aibo-symmetry-closure/`.
**Verdict: `MIRROR_EQUIVARIANCE_MANUFACTURES_SYMMETRY` — the central hypothesis is REFUTED.**

---

## The hypothesis (mine) and the test

Central hypothesis going in: *an equivariant policy PRESERVES a symmetry it is given but cannot
MANUFACTURE one the dynamics lacks*; so over the asymmetric diagonal-trot scaffold, forcing
equivariance would not yield a two-sided crab. The test: take the already-trained **one-sided** omni
MLP (+y reached, −y not), force it to be **exactly** mirror-equivariant by Reynolds symmetrization over
the order-2 mirror group (`a(s) = ½[f(s) + g·f(g·s)]`, `g` = swap-left/right + flip-lateral, an
involution), **no retraining**, and measure both sides.

## Result — REFUTED: equivariance manufactures the two-sided crab (over the *asymmetric* scaffold)

Per-goal min-distance (reached iff `< 0.12` and upright), diagonal scaffold, equivariance residual **0.0**:

| bearing | raw (trained) | mirror-equivariant |
|---:|---|---|
| 0° | 0.119 ✓ | 0.120 ✓ |
| +20° | 0.120 ✓ | 0.118 ✓ |
| **−20°** | **0.502 ✗** | **0.119 ✓** |
| +40° | 0.247 ✗ | 0.442 ✗ |
| −40° | 0.554 ✗ | 0.600 ✗ |
| **reach / +y / −y** | **0.40 / 1/2 / 0/2** | **0.60 / 1/2 / 1/2** |

The symmetrized policy **reaches −20°** (min_dist 0.502 → 0.119) that no trained policy ever reached,
and the two sides are now **perfectly matched** (+20° 0.118 ≈ −20° 0.119; +40° 0.442 ≈ −40° 0.600, both
just out of lateral range). It is a **two-sided symmetric crab**, obtained with **zero retraining** and
**exact** equivariance (residual 0.0). My hypothesis is **wrong**: over the asymmetric-looking diagonal
scaffold, hard equivariance **did** manufacture the symmetry.

## Why I was wrong — the diagonal trot is mirror-symmetric under a π phase-shift

The diagonal trot `(0,π,π,0)` is **not** instantaneously left-right symmetric, but it **is** symmetric
under swap-left/right **plus a half-period (π) phase-shift** (the full-stride symmetry). `mirror_obs`
already encodes that π-shift (`sin/cos(ph) *= −1`). So the mirrored −y recipe is the working +y recipe
**re-phased by π** — which over the diagonal trot lands on the *other* diagonal and produces a valid −y
crab. The **raw learned policy converged one-sided** (SAC found the +y optimum and never learned the −y
recipe); **forcing exact equivariance supplies the −y recipe for free** as the phase-shifted mirror of
the +y recipe.

This reconciles two earlier findings rather than contradicting them:
- **Phase A's constant-abduction probe** (a *fixed*, non-phase-shifted pattern) correctly showed the
  diagonal scaffold is asymmetric *for a static pattern* and the bound scaffold restores instantaneous
  antisymmetry. But the **phase-aware closed-loop** policy sees the diagonal trot's *stride* symmetry,
  which the static probe cannot. Both are true at their own timescale.
- **The failed mirror-augmentation** (soft, training-time data augmentation, degraded 3/5→2/5) vs. this
  **hard symmetrization** (architectural, exact): augmentation only *encourages* symmetry and SAC still
  converged asymmetric; the Reynolds average **enforces** it. The lever is HARD equivariance, not data.

## Bound scaffold (control)

The symmetrized policy over bound stays 0/2 both sides — bound barely locomotes (Phase A), so there is
nothing to symmetrize into a reach. Consistent with Phase A; not informative about the equivariance.

## Honest scope

One trained policy, one seed; ±40° stays out of range on both sides (a lateral-**range** limit, not a
symmetry limit — the symmetric crab reaches ±20° but not ±40°). The effect is large (−20° 0.502→0.119),
deterministic, and mechanistically explained (the π-phase-shift stride symmetry), so it is reported as a
verified positive, not a noisy A/B. Natural follow-up: **train** with the symmetrization in the loop (a
hard-equivariant actor) rather than wrapping a trained policy — expected to reach ±20° both sides
natively and possibly extend range.

## "HSiKAN?" — the equivariance principle is backbone-agnostic, but the *trained* HSiKAN has nothing to symmetrize

The wrap-and-symmetrize trick fixed the MLP. Does it fix the HSiKAN? Applying an exact mirror over the
**leg-hypergraph** obs (swap fl↔fr / bl↔br vertices + flip the lateral features, empirically pinned via
the +20°/−20° obs at a symmetric start) to the trained `signedkan` per-node omni policy: **no** — and an
**exhaustive search over every valid feature-flip mirror finds none** that makes it two-sided. But this
is **not** an equivariance failure. The diagnostic reveals why:

| policy | ‖raw action‖ | ‖mirror term‖ | ‖symmetrized‖ |
|---|---|---|---|
| MLP | **0.822** | 0.615 | 0.712 |
| signedkan | **0.066** | 0.067 | 0.060 |

The trained **signedkan learned a near-ZERO residual** (~0.066, an order of magnitude below the MLP's
active crab ~0.82) — it **rides the scaffold** instead of learning an active abduction crab. There is
almost nothing to symmetrize. Its +y-only reach is inherited from a **scaffold turning asymmetry**
(a=0 scaffold: +20° min_dist 0.141 vs −20° 0.221 — the known `SteeredTrotGait` one-way yaw bias), which
the tiny residual only bridges on the +y side; it is **not a learned crab at all**. This re-characterises
the earlier "HSiKAN ≈ MLP, both reach 0.40" result: they reach 0.40 by **different means** — the MLP by an
active symmetrizable crab, the HSiKAN by near-null scaffold-riding.

So the honest answer: (1) the mirror-equivariance **principle is backbone-agnostic** (`symmetrize` wraps
any policy; proven on the MLP); (2) it only helps a policy that **learned an active crab** to symmetrize;
(3) the trained HSiKAN did not, so for HSiKAN the lever is first to make it **use its abduction authority**
(exploration / training), then equivariance makes it two-sided — best done by **training a hard-equivariant
HSiKAN in the loop** (symmetrization inside the actor), the natural follow-up. The `Z_2` equivariance and
the sunflower `S_4` structural prior remain orthogonal and composable; neither manufactures a crab the
policy never learned.

## Files / tests

```
scenarios/aibo/mirror_equivariant.py           NEW  symmetrize() + equivariance_residual() (exact Z2 Reynolds average)
scenarios/aibo/run_aibo_mirror_equivariant.py  NEW  raw vs mirror-equivariant, diag + bound, per-goal
tests/test_aibo_mirror_equivariant.py          NEW  4 tests: mirror involutions, exact equivariance for arbitrary base, projection idempotence
reports/2026-07-28-aibo-residual-trot/result_mirror_equivariant.json  NEW
```

`ruff` clean; **111/111** AIBO tests green. CPU, seed 0, MuJoCo. CORE.YAML touched: none. The
"sunflower" connection: this exact `Z_2` equivariance composes with (is orthogonal to) the sunflower
`S_4` structural prior — the wrapper is backbone-agnostic; here it is applied to the flat MLP because
the one-sidedness (shown HSiKAN≈MLP) is not a backbone property.
