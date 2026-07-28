# Multi-seed correction — the recipe ordering holds, but the single-seed numbers were a lucky draw

**Date:** 2026-07-29 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION.** Corrects the single-seed claims in `2026-07-28-aibo-equivariant-train.md`.
**Verdict: `POST_HOC_BEST_BUT_NOT_GUARANTEED; IN_LOOP_NEVER_TWO_SIDED` (3 seeds).**

---

## Why

The 2026-07-28 conclusion ("post-hoc symmetrization 0.60 two-sided ≫ in-loop 0.20 symmetric-null") rested
on **one seed** — a violation of the repo's own rule against first-pass conclusions. This runs the three
recipes across **3 seeds** (0,1,2; 25k) and reports the distribution, not a point.

## Result (MLP omni crab, diag scaffold; +y/−y = goals reached out of 2 each)

| seed | raw (unconstrained) | post-hoc symmetrize | in-loop equivariant |
|---:|---|---|---|
| 0 | 0/0 (reach 0.2) | 0/0 (0.2) | 0/2 (0.4) |
| 1 | 1/2 (0.8) | 1/1 (0.6) | 0/1 (0.4) |
| 2 | 0/1 (0.4) | 2/1 (0.8) | 2/0 (0.6) |
| **two-sided seeds** | **1/3** | **2/3** | **0/3** |
| **median reach** | **0.40** | **0.60** | **0.40** |

## What holds, what does not

- **Holds:** the ordering **post-hoc ≥ raw ≈ in-loop** on median reach, and — the robust part — **in-loop
  equivariance NEVER reaches both sides (0/3)**. Imposing symmetry during training does not yield a
  two-sided crab at any seed. That single-seed claim survives multi-seed.
- **Corrected / weaker than claimed:** post-hoc is **2/3, not a guarantee**. On seed 0 the *unconstrained*
  policy discovered **no** crab (raw 0/0), so post-hoc had nothing to mirror (0/0). Post-hoc only wins
  **when the free policy first discovers the crab** — which is itself seed-fragile. The "0.60 two-sided"
  headline was a lucky seed; the honest statement is "post-hoc is the best available recipe and works on a
  majority of seeds, conditional on discovery."
- **Note on the per-seed one-sidedness of the (exactly-equivariant) in-loop policy** (e.g. seed 2 `2/0`):
  the eval gives each goal its own start seed (`500+i`), so a symmetric policy can clear one goal's start
  and miss the mirror goal's harder start — the asymmetry is in the *evaluation starts*, not the policy.
  The `two_sided` metric (needs ≥1 on each side) still correctly captures "does not robustly reach both".

## Honest caveats (the corner I cut)

- **25k, not the 30k** the original used — to save wall-clock. Seed 0's raw finding nothing is partly
  undertraining; at 30k discovery is likelier. So this **understates** raw/post-hoc a little and does not
  cleanly isolate the recipe from the training budget. The **in-loop 0/3** conclusion is budget-robust
  (it fails regardless), but the post-hoc rate would firm up at 30k / more seeds.
- **3 seeds** shows variance but is below the repo's 5-iteration bar; 5 seeds at 30k is the clean rerun.

## Clean rerun (5 seeds, 30k) — walks the advantage back further

The non-corner-cut rerun (seeds 0-4, 30k, the tested budget):

| recipe | two-sided seeds | median reach |
|---|---|---|
| raw | **0/5** | 0.40 |
| post-hoc symmetrize | **2/5** | 0.40 |
| in-loop equivariant | **1/5** | 0.40 |

**All three recipes have the SAME median reach (0.40).** The symmetry manipulations do **not** improve the
overall goal-reach rate — they only shuffle *which* goals are hit. Post-hoc's edge is **marginal** (2/5 vs
1/5 two-sided), and the 3-seed "in-loop 0/3, post-hoc 0.60" was noise: at 5 seeds in-loop reaches both on
1 seed and post-hoc's median is 0.40, not 0.60. So the honest, final statement is: **post-hoc
symmetrization gives at most a small, unreliable improvement in reaching both lateral sides, and changes
the overall goal-reach rate essentially not at all.**

## Bottom line (superseded twice — the honest end)

The single-seed headline (post-hoc 0.60 ≫ in-loop 0.20) did not survive multi-seed. At 5 seeds / 30k the
recipes are **tied on goal-reach (all 0.40)**; post-hoc's only, marginal, benefit is a slightly higher
chance of reaching *both* lateral sides (2/5). The crab-symmetry axis, rigorously measured, **barely moves
the actual objective (reaching the target)** — which is the empirical case for refocusing off crab-symmetry
and onto the goal-reaching bottleneck (turning / heading control) directly.

## Files

```
scenarios/aibo/run_aibo_symmetry_multiseed.py  NEW  3-recipe × N-seed harness (raw / post-hoc / in-loop)
reports/2026-07-28-aibo-residual-trot/result_symmetry_multiseed.json  NEW
```

`ruff` clean. CPU, seeds 0-2, 25k. CORE.YAML: none. SIMULATION.
