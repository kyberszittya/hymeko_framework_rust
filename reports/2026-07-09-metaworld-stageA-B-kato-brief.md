# MetaWorld reward-ablation Stage A→B — executive brief (Ito+Kato)

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration` · closure commit `a92a8c6`
**Audience:** Kato / Ito. Read-only consolidation of the Stage A→B arc. **No new experiments were run for this
brief.**

---

## Headline

Using the HyMeKo `.hymeko` reward as the single source of truth, we ablated one declared reward term
(`mw_in_place`) on MetaWorld pick-place and traced the consequence through two levels. At the
**reward-computation level** (fixed rollouts, no training) `mw_in_place` is the dominant reward driver — ablating
it collapses its CIP/LiNGAM-SH mechanism loading (1918→373, 5.1×) and doubles the reward↔monitor disagreement,
robust across 5 seeds; `mw_grasp` is inert (a clean negative control) and `mw_dist` is a weak secondary. At the
**policy-learning level** (BC warm-start + reward fine-tune, single seed), from an *identical* cloned start,
training under the full reward reaches 62.5% success while training under `mw_in_place_off` collapses to **0%** —
the policy never grasps and barely approaches the object. `mw_in_place` is load-bearing both in what the reward
*computes* and in what a policy trained on it can *achieve*.

## Stage A — reward-computation level (no training)

The HyMeKo reward is a declared `Σ weight·term`. Ablating a term = zeroing its weight and recomputing the reward
offline on fixed scripted rollouts, then re-running CIP/DirectLiNGAM + weighted LiNGAM-SH + cross-view. Multi-seed
(5 batches × N=80):

| ablation | verdict (5 seeds) | reward change | dropped-term loading | reward↔monitor disagreement |
|---|---|---:|---|---:|
| `mw_grasp_off` (negative control) | **NOT_SUPPORTED 5/5** | 0.061 | 31.6 → 22.5 (inert) | 0.061 (no consistent move) |
| `mw_dist_off` (secondary) | SECONDARY 2 / NOT_SUPPORTED 3 | 0.124 | obj_to_target collapse noisy | 0.071 |
| `mw_in_place_off` (positive control) | **SUPPORTED 5/5** | 1.044 | progress **1918 → 373** (5.1×) | **0.080 → 0.268** (spike) |

- **Ordering is stable:** `in_place (1.04) ≫ dist (0.12) > grasp (0.06)`, tracking fitted weights `8.5 ≫ 4.8 > 0.08`.
- **Cross-view verification: 100 % (25/25 conditions).**
- Interpretation: `mw_grasp` is a genuine negative (a minor term the pipeline correctly finds inert); `mw_in_place`
  is a genuine positive (dominant, collapses sharply); `mw_dist` is a real-but-weak secondary — reported, not
  overclaimed.

## Stage B — policy-learning level (BC warm-start + reward fine-tune, 1 seed)

From the **same** BC-cloned, reward-agnostic base policy (fair shared start), fine-tune REINFORCE under each reward:

| trained under | success | grasp | near (approaches object) | reward↔monitor disagreement | reward under TRUE reward |
|---|---:|---:|---:|---:|---:|
| **original** | **0.625** | 0.494 | 0.555 | 0.145 | **+749** |
| **`mw_in_place_off`** | **0.000** | 0.000 | 0.010 | 0.312 | **−218** |
| Δ (off − original) | −0.625 | −0.494 | −0.545 | +0.167 | — |

**Policy-learning verdict: SUPPORTED (single seed).** Removing `mw_in_place` does not merely lower performance — it
collapses the policy: no grasp, almost no approach, 0% delivery, and reward↔monitor disagreement doubles. The same
start under the full reward learns to 62.5%.

## The observation-normalization correction (fairness)

The first BC clone rolled out **0%** success despite a low BC loss — matching the repo's own note that MetaWorld
pick-place BC "rolls out 0%" (covariate shift). Rather than accept that as a task wall, we ran the discriminating
check: MetaWorld observations are **unnormalized**, so a raw-obs tanh-MLP barely responds. Adding input
standardization (mean/std fit from the demos) took **BC from 0.00 → 0.95** success.

**Why this matters for Kato:** the Stage-B contrast is only meaningful if both arms start from a *competent* policy.
The fix means the 62.5% vs 0% divergence is a real reward-driven effect from a working base — not an artifact of a
dead clone. The "covariate-shift wall" was a normalization bug, not a fundamental limit.

## Honest caveats

- **Stage B is single-seed.** The effect is large and mechanistically coherent, but not yet median/IQR. Env
  randomization is seed-uncontrolled (BC-eval success itself varies 0.46–0.95 run-to-run) — which is exactly why
  the two arms share **one** BC start: the comparison is within-run controlled.
- **REINFORCE is a bounded smoke optimizer, not the final RL algorithm.** The result is the *contrast* under a
  fair, shared, bounded protocol — not an SOTA absolute pick-place number. A stronger optimizer (PPO) would raise
  both ceilings; the ablation contrast is the claim, not the 62.5%.
- **Reward-computation → policy-learning is a bridge, not a substitution.** Stage B is a policy-level claim
  consistent with Stage A, on one task (pick-place), one seed.

## Claim discipline (three tiers)

**A — safe to claim now:**
- The HyMeKo `.hymeko` reward SoT can drive reward-term ablation (deterministic, non-mutating).
- Runtime monitors expose reward/task disagreement (reward↔monitor concordance).
- LiNGAM-SH mechanism grouping can be intervened on (ablate a term → re-fit → cross-view).
- `mw_in_place` is load-bearing **in reward computation** (5-seed SUPPORTED) **and in a single-seed policy-learning
  smoke** (62.5% → 0%).

**B — claim only with the caveat attached:**
- The Stage-B policy consequence is supported **but single-seed**.
- The REINFORCE result shows a **contrast**, not final-optimizer performance.

**C — do not claim yet:**
- Multi-seed policy-learning robustness (not run).
- PPO-level / SOTA performance (not run).
- Generalization across all MetaWorld tasks (only pick-place studied).
- Theorem-level LiNGAM-SH identifiability (empirical mechanism grouping, not an identifiability proof).

## Next gated decisions (nothing runs automatically)

1. **Multi-seed Stage B** (3–5 seeds → median/IQR) to upgrade the policy-level claim from "supported, single-seed"
   to a robust published result. Larger compute; requires go-ahead.
2. **PPO instead of REINFORCE** to raise both arms' ceilings and show the contrast survives a stronger optimizer.
   Larger compute; requires go-ahead.

Neither is run here. Artifact map: `reports/2026-07-09-metaworld-stageA-B-artifact-index.md`.
