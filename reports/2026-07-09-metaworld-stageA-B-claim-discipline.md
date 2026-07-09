# MetaWorld reward-ablation Stage A→B — claim discipline

**Date:** 2026-07-09 · closure commit `a92a8c6` · **UPDATED after 5-seed Stage B** (see
[multiseed report](2026-07-09-metaworld-stageb-multiseed-result.md))
One page. What we may say, what we must qualify, and what we must not say yet about the Stage A→B result. Pair with
the executive brief ([reports/2026-07-09-metaworld-stageA-B-kato-brief.md](2026-07-09-metaworld-stageA-B-kato-brief.md))
and the artifact index ([reports/2026-07-09-metaworld-stageA-B-artifact-index.md](2026-07-09-metaworld-stageA-B-artifact-index.md)).

> **⚠️ MULTI-SEED + PPO CORRECTION.** The single-seed Stage-B *success* collapse (62.5%→0%) is **refuted**: 5-seed
> REINFORCE is not robust (reverses on seed 4), and **5-seed PPO shows both profiles reach ~100% success**
> (contrast median 0.0) — a BC-anchored policy is not disabled by the ablation under a stable optimizer. What
> survives is that **reward↔monitor disagreement is higher under `mw_in_place_off`** (REINFORCE 5/5, PPO 4/5,
> ~2.3× median). The tiers below reflect this. Details:
> [PPO report](2026-07-09-metaworld-stageb-ppo-multiseed.md).

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
5. **Under `mw_in_place_off`, a trained policy's reward↔monitor disagreement is higher** (REINFORCE **5/5**, PPO
   **4/5**, ~2.3× median) — the reward stops tracking the task even when the policy *succeeds* (PPO). The
   reward-computation signature survives into training regardless of policy competence.

## B — Claim only with the caveat attached

State the number and the caveat in the same breath; never quote one without the other.

1. **`mw_dist` is a weak secondary driver, not a positive.** Its reward-change is consistent (~0.12) but its
   loading collapse is noisy (SECONDARY 2 / NOT_SUPPORTED 3) — report as secondary, do not promote.

## C — Do not claim (multi-seed + PPO refuted the strong form)

1. **A policy-learning *success* effect of the ablation.** ❌ REFUTED under **both** optimizers — 5-seed REINFORCE
   is not robust (reverses on seed 4) and 5-seed PPO gives **both profiles ~100% success** (contrast median 0.0).
   Do **not** say "training without `mw_in_place` collapses/changes the policy's success." A BC-anchored policy is
   not disabled by the ablation.
2. **That `mw_in_place` is needed to *learn* pick-place.** OPEN — from-scratch RL was attempted and is **invalid
   as a reward test**: sanity diagnostics show the from-scratch PPO cannot learn even the trivial *reach* reward at
   its default exploration setting, while the harness/metrics/control are provably correct (scripted reach fires
   `near` 100 %, BC succeeds 0.94). The 0%-vs-0% measured **PPO-setup inadequacy** (diagnosis B, fixable via
   exploration-std/budget), not the reward. A valid test first needs an optimizer that learns reach→grasp from
   scratch (tuned PPO or SAC). Do not claim either way. See
   [sanity diagnostics](2026-07-09-pick-place-from-scratch-sanity-diagnostics.md).
3. **Generalization across MetaWorld tasks.** Only pick-place was studied (coffee-push never grasps, so `mw_grasp`
   was not even in-frame there).
4. **Theorem-level LiNGAM-SH identifiability.** The mechanism grouping is an empirical, cross-view-checked
   construction, not an identifiability proof. No formal-guarantee language.

## The single sentence to lead with

> Using the HyMeKo reward as the source of truth, ablating the `mw_in_place` term collapses its LiNGAM-SH mechanism
> loading (5-seed robust) and, in a 5-seed BC-anchored policy study under two optimizers, raises a trained policy's
> reward↔monitor disagreement (~2.3×, 4–5/5 seeds) **even when the policy still succeeds** — so `mw_in_place` is
> load-bearing in the reward computation and disrupts reward↔task alignment, while task *success* is unaffected in
> a BC-anchored policy (the ablation does not disable a policy that already knows the task).
