# Stage A→B→repair — final synthesis (MetaWorld pick-place, HyMeKo reward audit & repair)

**Date:** 2026-07-10 · Aiko · branch `hymeko-neuro-migration` · **FROZEN.**
One-page synthesis of the whole arc for Kato / Ito and later paper integration. No experiments run for this doc.

---

## The arc in one line

Using the HyMeKo reward as a source of truth and its runtime monitors as semantic instruments, we **audited** a
MetaWorld pick-place reward (found a load-bearing progress proxy and a farmable proxy-exploit) and then **repaired**
it into a monitor-aligned reward that suppresses proxy farming ≈45× while staying dense — verified offline and by
cross-view HyMeKo checks. Policy-learning consequences were probed but are deliberately **not** over-claimed.

## What holds, what is qualified, what is out of scope

| level | finding | status |
|---|---|---|
| **reward computation (Stage A)** | `mw_in_place` is the dominant reward driver; ablation re-parents toward delivery; disagreement spikes | **robust, 5-seed** |
| **policy fine-tune from BC** | ablation does not change task *success* (REINFORCE not robust; PPO both ~100%); reward↔monitor disagreement higher under the ablated reward (4–5/5) | success: no robust effect · disagreement: robust-ish |
| **from-scratch learning** | 0%-vs-0% was a **PPO-setup** issue (harness/metrics proven correct), not the reward; optimizer partly repaired (3 real bugs) but reach not robust | **inconclusive / gated** |
| **reward repair (monitor-aligned)** | reduces reward↔monitor disagreement, stays dense, cross-view ✅; **suppresses proxy farming ≈45×** on decoupled trajectories; does not destroy a BC-anchored policy (1 seed) | **supported** |

## The one master claim-discipline table

| stage | question | result | valid claim | caveat |
|---|---|---|---|---|
| reward-computation ablation | Is `mw_in_place` load-bearing? | SUPPORTED 5-seed | dominates the reward mechanism | reward-computation level |
| BC-anchored fine-tune | Does ablation change success? | not robust / ~equal under PPO | success unaffected in BC-anchored policy | single→5-seed; PPO both succeed |
| from-scratch diagnostics | reward or setup? | PPO-setup issue | not a reward result, not a wall | harness/metrics proven correct |
| optimizer repair | can PPO learn reach? | 3 bugs fixed, reach not robust | reach sometimes learned | ablation still invalid from scratch |
| monitor-aligned repair | reduce disagreement, keep dense? | R1–R3 pass | monitor-aligned reward reduces disagreement, stays dense | R2 collinear; R3 single-seed |
| anti-farming validation | suppress proxy farming? | YES ≈45×, HEALTHY 5/5 | repaired reward suppresses proxy farming | counterfactual trajectories, not a learned adversary |

## The headline result (for slides)

On eight decoupled counterfactual trajectories, the **original** reward scores a pure *proxy-exploit* (contact-like
evidence, **zero delivery**) at 88.8 — as high as true delivery (86.8) and 82 % of success. The **monitor-aligned**
reward scores the same proxy at 2.0 → **proxy/success 0.816 → 0.018, ≈45× suppression**, while preserving dense
pre-success shaping (nonzero pre-success fraction 0.95–1.0 on approach/delivery). During the repair we also **caught
a farming vector in our own fix** (holding the object aloft farmed lift reward) and corrected it with a
potential-based lift — a reminder that reward repairs must themselves be audited.

## What this supports for HyMeKo

HyMeKo runtime monitors are usable not only to **audit** learned policies (post-hoc success/consistency checking)
but also as semantic instruments to **repair reward functions** — gating proxy terms through task-progress
predicates, with the repair verified by offline recomputation and cross-view consistency against the HyMeKo
representation.

## Report index

| topic | report |
|---|---|
| reward SoT → LiNGAM-SH | `reports/2026-07-09-metaworld-reward-sot-lingam-sh-integration.md` |
| Stage A ablation + positive control + multi-seed | `reports/2026-07-09-metaworld-reward-ablation-{stageA,positive-control,multiseed}.md` |
| Stage B setup / result / multi-seed / PPO | `reports/2026-07-09-metaworld-stageb-*.md` |
| Kato brief / artifact index / claim discipline | `reports/2026-07-09-metaworld-stageA-B-*.md` |
| from-scratch + sanity diagnostics + optimizer repair | `reports/2026-07-{09,10}-*from-scratch*.md`, `…-ppo-optimizer-repair.md` |
| **monitor-aligned repair + anti-farming + final synthesis** | `reports/2026-07-10-*monitor-aligned*.md` |

**Recommendation: stop the current arc here.** Frozen, verified in both directions, Kato-ready. Remaining levers
(adversarial-policy farm search, multi-seed R3, SAC/curriculum from-scratch) are larger, gated follow-ons.
