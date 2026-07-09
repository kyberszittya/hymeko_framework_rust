# MetaWorld reward-ablation Stage A→B — claim discipline

**Date:** 2026-07-09 · closure commit `a92a8c6` · **UPDATED after 5-seed Stage B** (see
[multiseed report](2026-07-09-metaworld-stageb-multiseed-result.md))
One page. What we may say, what we must qualify, and what we must not say yet about the Stage A→B result. Pair with
the executive brief ([reports/2026-07-09-metaworld-stageA-B-kato-brief.md](2026-07-09-metaworld-stageA-B-kato-brief.md))
and the artifact index ([reports/2026-07-09-metaworld-stageA-B-artifact-index.md](2026-07-09-metaworld-stageA-B-artifact-index.md)).

> **⚠️ MULTI-SEED CORRECTION (5 seeds):** the single-seed Stage-B *success* collapse (62.5%→0%) is **NOT robust** —
> it reverses on seed 4 (off 1.00 vs original 0.29) and the success axis is dominated by BC + REINFORCE variance.
> What IS robust across 5/5 seeds is that **reward↔monitor disagreement is higher under `mw_in_place_off`**
> (0.254 vs 0.087). The tiers below reflect this correction.

---

## A — Safe to claim now

Each backed by a committed artifact + passing tests.

1. **The HyMeKo `.hymeko` reward is a usable single source of truth for reward-term ablation.** Ablation is
   deterministic and does not mutate the declaration; fidelity to the env reward is measured (R² ≈ 0.88–0.92).
2. **Runtime monitors expose reward↔task disagreement.** The RewardConsistencyMonitor quantifies
   reward/monitor concordance, and it moves in the predicted direction under ablation.
3. **LiNGAM-SH mechanism grouping is intervenable.** Drop a declared term → recompute reward offline → re-fit the
   mechanism → cross-view verify. Cross-view held 25/25 across the multi-seed panel.
4. **`mw_in_place` is load-bearing in reward computation** (**5-seed SUPPORTED**: loading 1918→373, disagreement
   0.080→0.268). `mw_grasp` is a confirmed inert negative control (NOT_SUPPORTED 5/5).
5. **Under `mw_in_place_off`, a trained policy's reward↔monitor disagreement is robustly higher** (**5/5 seeds**,
   median 0.254 vs 0.087, non-overlapping) — the reward stops tracking the task even when the policy occasionally
   succeeds. The reward-computation signature survives into training.

## B — Claim only with the caveat attached

State the number and the caveat in the same breath; never quote one without the other.

1. **The Stage-B success median leans toward the original reward — but the contrast is not robust.** "Across 5
   seeds the median success is 0.29 (original) vs 0.04 (off), but IQRs overlap and one seed reverses" — never quote
   the median without the non-robustness.
2. **`mw_dist` is a weak secondary driver, not a positive.** Its reward-change is consistent (~0.12) but its
   loading collapse is noisy (SECONDARY 2 / NOT_SUPPORTED 3) — report as secondary, do not promote.

## C — Do not claim yet

Not run, or the multi-seed pass has now refuted the strong form.

1. **A robust policy-learning *success* collapse.** ❌ REFUTED at this optimizer/budget — the 62.5%→0% was a
   favorable seed; 5-seed success is not robust (reverses on seed 4). Do **not** say "training without
   `mw_in_place` collapses the policy." A stronger optimizer might change this, but as run it is not supported.
2. **PPO-level / SOTA performance.** Only a bounded REINFORCE smoke was run. No absolute performance claim.
3. **Generalization across MetaWorld tasks.** Only pick-place was studied. No cross-task claim (coffee-push never
   grasps, so `mw_grasp` was not even in-frame there).
4. **Theorem-level LiNGAM-SH identifiability.** The mechanism grouping is an empirical, cross-view-checked
   construction, not an identifiability proof. No formal-guarantee language.

## The single sentence to lead with

> Using the HyMeKo reward as the source of truth, ablating the `mw_in_place` term collapses its LiNGAM-SH mechanism
> loading (5-seed robust) and, in a 5-seed BC-anchored policy-learning study, robustly raises a trained policy's
> reward↔monitor disagreement (5/5 seeds) — evidence that `mw_in_place` is load-bearing at the reward-computation
> level and disrupts reward↔task alignment at the policy level; the raw *success* contrast is not robust at this
> (REINFORCE) optimizer and needs a stronger optimizer to settle.
