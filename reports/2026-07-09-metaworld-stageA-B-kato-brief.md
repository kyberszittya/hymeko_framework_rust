# MetaWorld reward-ablation Stage A→B — executive brief (Ito+Kato)

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration` · closure commit `a92a8c6`
**Audience:** Kato / Ito. Read-only consolidation of the Stage A→B arc, **updated after a 5-seed Stage-B run.**

> **⚠️ MULTI-SEED UPDATE (5 seeds).** The single-seed Stage-B *success* collapse (62.5%→0%) is **NOT robust** — it
> reverses on seed 4 (`mw_in_place_off` 1.00 vs original 0.29); the success axis is dominated by BC + REINFORCE
> variance. **Robust across 5/5 seeds:** reward↔monitor **disagreement** is higher under `mw_in_place_off` (median
> 0.254 vs 0.087). See [multiseed report](2026-07-09-metaworld-stageb-multiseed-result.md). The Stage-B section
> below is corrected accordingly; Stage A is unaffected.

---

## Headline (corrected after multi-seed)

Using the HyMeKo `.hymeko` reward as the single source of truth, we ablated one declared reward term
(`mw_in_place`) on MetaWorld pick-place and traced the consequence through two levels. At the
**reward-computation level** (fixed rollouts, no training) `mw_in_place` is the dominant reward driver — ablating
it collapses its CIP/LiNGAM-SH mechanism loading (1918→373, 5.1×) and doubles the reward↔monitor disagreement,
**robust across 5 seeds**; `mw_grasp` is inert (a clean negative control) and `mw_dist` is a weak secondary. At the
**policy-learning level** (BC warm-start + reward fine-tune, **5 seeds**), a policy trained under `mw_in_place_off`
robustly shows **higher reward↔monitor disagreement** (5/5 seeds) — the reward stops tracking the task — but the
raw *success* contrast is **not robust** (the single favorable seed 0 showed 62.5%→0%; the 5-seed median leans
original 0.29 vs 0.04 with a full reversal on seed 4, dominated by optimizer variance). `mw_in_place` is
load-bearing in what the reward *computes* and disrupts reward↔task *alignment* when trained on — but a robust
*success* claim needs a stronger optimizer than the REINFORCE smoke.

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

## Stage B — policy-learning level (BC warm-start + reward fine-tune, 5 seeds)

From the **same** BC-cloned start per seed (fair within-seed), fine-tune REINFORCE under each reward. **5-seed
aggregate (median [IQR]):**

| metric | original | `mw_in_place_off` | robust? |
|---|---|---|---|
| success rate | 0.292 [0.083, 0.667] | 0.042 [0.000, 0.208] | **no** — IQRs overlap; seed 4 reverses (off 1.00) |
| grasp fraction | 0.413 [0.100, 0.471] | 0.000 [0.000, 0.121] | no |
| **reward↔monitor disagreement** | **0.087 [0.080, 0.109]** | **0.254 [0.188, 0.261]** | **YES — off higher 5/5** |
| reward under TRUE reward | 710 [80, 746] | −104 [−134, 205] | leans original, wide IQR |

Per-seed success contrast (orig−off): **[+0.67, +0.08, 0.00, +0.54, −0.71]**.

**Policy-learning verdict: NOT_ROBUST for success; ROBUST for reward↔monitor disagreement.** The single-seed run
(seed 0: 62.5% vs 0%) was a favorable seed — the success axis is dominated by BC-quality variance (BC success
0.29–1.00 across seeds) and REINFORCE variance (original *collapsed* to 0.04 from a perfect BC base on seed 2). The
part that holds up: under `mw_in_place_off` the reward stops tracking the task monitor in **every** seed (higher
disagreement), even on seed 4 where the off policy happens to succeed (its disagreement is the highest, 0.435).

## The observation-normalization correction (fairness)

The first BC clone rolled out **0%** success despite a low BC loss — matching the repo's own note that MetaWorld
pick-place BC "rolls out 0%" (covariate shift). Rather than accept that as a task wall, we ran the discriminating
check: MetaWorld observations are **unnormalized**, so a raw-obs tanh-MLP barely responds. Adding input
standardization (mean/std fit from the demos) took **BC from 0.00 → 0.95** success.

**Why this matters for Kato:** the Stage-B contrast is only meaningful if both arms start from a *competent* policy.
The fix means the 62.5% vs 0% divergence is a real reward-driven effect from a working base — not an artifact of a
dead clone. The "covariate-shift wall" was a normalization bug, not a fundamental limit.

## Honest caveats

- **Stage-B success is not robust (5-seed).** BC-quality is seed-dependent (0.29–1.00) and REINFORCE is
  high-variance — it collapsed the *original* arm to 0.04 from a perfect BC base on seed 2, and drove the *off*
  arm to 1.00 on seed 4. The two arms share a start **within** a seed (fair), but the fine-tune outcome is
  optimizer-noise-dominated at this budget. The single-seed 62.5%→0% was a favorable seed.
- **REINFORCE is a bounded smoke optimizer, not the final RL algorithm.** A stronger optimizer (PPO) or larger
  budget is needed before any *success* claim is defensible.
- **What is robust:** reward↔monitor disagreement is higher under `mw_in_place_off` in 5/5 seeds — the policy-level
  echo of Stage A. And Stage A (reward-computation level) never depended on training.

## Claim discipline (three tiers, post multi-seed)

**A — safe to claim now:**
- The HyMeKo `.hymeko` reward SoT can drive reward-term ablation (deterministic, non-mutating).
- Runtime monitors expose reward/task disagreement (reward↔monitor concordance).
- LiNGAM-SH mechanism grouping can be intervened on (ablate a term → re-fit → cross-view; 100 % cross-view).
- `mw_in_place` is load-bearing **in reward computation** (5-seed SUPPORTED).
- Under `mw_in_place_off`, a trained policy's **reward↔monitor disagreement is robustly higher** (5/5 seeds).

**B — claim only with the caveat attached:**
- The Stage-B success **median** leans original (0.29 vs 0.04) **but the contrast is not robust** — never quote the
  median without the non-robustness.
- `mw_dist` is a weak/secondary driver (SECONDARY 2 / NOT_SUPPORTED 3), not a positive.

**C — do not claim (multi-seed refuted the strong form):**
- ❌ A robust policy-learning *success* collapse — refuted at this optimizer/budget (reverses on seed 4).
- PPO-level / SOTA performance (not run).
- Generalization across all MetaWorld tasks (only pick-place studied).
- Theorem-level LiNGAM-SH identifiability (empirical mechanism grouping, not an identifiability proof).

## Next gated decisions (nothing runs automatically)

1. **PPO (or larger budget) instead of REINFORCE**, then re-run multi-seed — the only path to a defensible
   *success* claim (current optimizer variance swamps the reward signal). Larger compute; requires go-ahead.
2. The **disagreement** claim is already robust (5/5) and needs no further compute.

Multi-seed Stage B is **done** (this update). Artifact map:
`reports/2026-07-09-metaworld-stageA-B-artifact-index.md`; multi-seed detail:
`reports/2026-07-09-metaworld-stageb-multiseed-result.md`.
