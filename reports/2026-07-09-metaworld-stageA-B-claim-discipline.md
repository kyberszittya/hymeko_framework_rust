# MetaWorld reward-ablation Stage A→B — claim discipline

**Date:** 2026-07-09 · closure commit `a92a8c6`
One page. What we may say, what we must qualify, and what we must not say yet about the Stage A→B result. Pair with
the executive brief ([reports/2026-07-09-metaworld-stageA-B-kato-brief.md](2026-07-09-metaworld-stageA-B-kato-brief.md))
and the artifact index ([reports/2026-07-09-metaworld-stageA-B-artifact-index.md](2026-07-09-metaworld-stageA-B-artifact-index.md)).

---

## A — Safe to claim now

Each backed by a committed artifact + passing tests.

1. **The HyMeKo `.hymeko` reward is a usable single source of truth for reward-term ablation.** Ablation is
   deterministic and does not mutate the declaration; fidelity to the env reward is measured (R² ≈ 0.88–0.92).
2. **Runtime monitors expose reward↔task disagreement.** The RewardConsistencyMonitor quantifies
   reward/monitor concordance, and it moves in the predicted direction under ablation.
3. **LiNGAM-SH mechanism grouping is intervenable.** Drop a declared term → recompute reward offline → re-fit the
   mechanism → cross-view verify. Cross-view held 25/25 across the multi-seed panel.
4. **`mw_in_place` is load-bearing** — in reward computation (**5-seed SUPPORTED**: loading 1918→373, disagreement
   0.080→0.268) **and** in a single-seed policy-learning smoke (**62.5% → 0%** success from an identical BC start).
   `mw_grasp` is a confirmed inert negative control (NOT_SUPPORTED 5/5).

## B — Claim only with the caveat attached

State the number and the caveat in the same breath; never quote one without the other.

1. **The Stage-B policy consequence is supported — but single-seed.** "Training without `mw_in_place` collapses the
   policy to 0% (single seed; multi-seed not yet run)." Env randomization makes even BC-eval vary run-to-run
   (0.46–0.95), which is why the arms share one controlled BC start.
2. **The REINFORCE result is a contrast, not final-optimizer performance.** "62.5% vs 0% under a fair, shared,
   bounded REINFORCE protocol" — not "our policy achieves 62.5% on pick-place." A stronger optimizer (PPO) would
   raise both ceilings.
3. **`mw_dist` is a weak secondary driver, not a positive.** Its reward-change is consistent (~0.12) but its
   loading collapse is noisy (SECONDARY 2 / NOT_SUPPORTED 3) — report as secondary, do not promote.

## C — Do not claim yet

Not run, or not the kind of statement this evidence supports.

1. **Multi-seed policy-learning robustness.** Stage B is one seed. No median/IQR policy-level claim.
2. **PPO-level / SOTA performance.** Only a bounded REINFORCE smoke was run. No absolute performance claim.
3. **Generalization across MetaWorld tasks.** Only pick-place was studied. No cross-task claim (coffee-push never
   grasps, so `mw_grasp` was not even in-frame there).
4. **Theorem-level LiNGAM-SH identifiability.** The mechanism grouping is an empirical, cross-view-checked
   construction, not an identifiability proof. No formal-guarantee language.

## The single sentence to lead with

> Using the HyMeKo reward as the source of truth, ablating the `mw_in_place` term collapses its LiNGAM-SH mechanism
> loading (5-seed robust) and, in a single-seed BC-anchored policy-learning smoke, collapses a trained policy from
> 62.5% to 0% success — evidence that `mw_in_place` is load-bearing at both the reward-computation and
> policy-learning levels, pending multi-seed confirmation.
