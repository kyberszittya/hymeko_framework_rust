---
campaign: COIN contact-stratified replay experiment (CONTROL vs STRATIFIED)
title: Does replay sampling alone generalize the certified bilateral delivery mechanism?
date: 2026-07-20
branch: exp/coin-contact-stratified-replay
source_commit: 37d923a
verdict: NEGATIVE on seed 0 only — OVERTURNED by the seeds 0-3 replication (SEED_SENSITIVE); see 2026-07-20-coin-contact-replay-multiseed.md
---

> **CORRECTION (2026-07-20 21:25, replication):** this single-seed NEGATIVE did **not** replicate. Seeds 0–3 give paired
> Δstrict = [−1, +2, +1, −2] (median 0, IQR 2.5, 2 pos / 2 negative) → **SEED_SENSITIVE**, not NEGATIVE. The seed-0 result
> was one draw from a high-variance distribution. The §13 "specify 2-actor × 2-critic" recommendation below is therefore
> **withdrawn** (it was gated on CONFIRMED_NEGATIVE/NO_EFFECT). See `2026-07-20-coin-contact-replay-multiseed.md`.

# Contact-stratified replay — matched CONTROL vs STRATIFIED

**Created-at:** 2026-07-20 20:40 JST. Single-variable experiment: the replay **sampler** only. Both arms continue
50,000 steps from `sac_actor_best.pt` (the reproducible-certified checkpoint) with **identical** env / K0 / obs schema /
action semantics / delivery-v2b reward / strict predicate / actor+critic architecture / SAC hyperparameters / BC
competence gate / state splits / eval path / seed. Config diff = sampler fields only (`launch_manifest.json`).

## Verdict: **NEGATIVE**

Contact-stratified replay **reduced** certified competence relative to the uniform control and **lost** the previously
reproducible state-64102 delivery. This is a single matched seed, but the degradation is **internally consistent across
five endpoints** (strict coverage, P(attr|zone), P(clean|zone), P(bilateral|zone), 64102 retention, attribution margin),
so it is more than a single noisy number — while still provisional pending a multi-seed replication.

## The replay-sampler extension (canonical, in place)

`ReplayBuffer` gained a per-transition provenance tag (0 = ONLINE default → uniform path byte-unchanged) + a
`sample_stratified(demo_frac, strata_weights, shortage, account)` that composes `demo_frac` demos (split by contact
strata) + the rest ONLINE; thin strata sample with replacement (logged), empty strata redistribute (never silently
filled from approach), batch size exact, deterministic. `train_sac` uses it when `demo_frac_fn`+`strata_weights` are
supplied, else the unchanged `buf.sample`. 8 regression tests. **Preflight:** corpus 5957, empirical demo_frac 0.5 and
per-stratum 35/25/15/15/10 hit, no shortage.

## Corpus (A1/A4, labelled by contact quality via the public rollout trace)

| stratum | count | definition |
|---|---|---|
| CERTIFIED_BILATERAL | 351 | strict, or zone + attribution≥0.60 + body≤0.20 + clean + bilateral |
| HIGH_QUALITY_CONTACT | 1138 | attribution≥0.60 + body≤0.20 + clean, not certified |
| RECOVERY | 1123 | loss-of-both-contact → recontact window (A4) |
| CONTRASTIVE_BULLDOZE | 1097 | zone + (attribution<0.60 or ¬clean or ¬bilateral) — the near-strict failures |
| GENERAL_PROGRESS | 2248 | valid target-directed progress, none of the above |

STRATIFIED demo mix: 35 % CERTIFIED / 25 % HIGH_QUALITY / 15 % RECOVERY / 15 % CONTRASTIVE / 10 % GENERAL; demo/online
50 %→25 % at 25k (never <25 %). All realized exactly (`train.log` `[replay]` lines; `comparison.json`).

## Best-checkpoint comparison on the 18-state eval set (4 DEMO + 14 VAL)

| endpoint | CONTROL (best @35k) | STRATIFIED (best @42.5k) |
|---|---|---|
| **strict count / coverage** | **4 / 4** | 3 / 3 |
| strict states | 64102, 64111, 64201, 64203 | 64111, 64113, 64201 |
| loose zone rate | 0.50 | 0.44 |
| **P(fingertip-attr ≥ 0.60 \| zone)** | **0.67** | 0.50 |
| P(clean mechanism \| zone) | 0.44 | 0.38 |
| P(bilateral contact \| zone) | 0.56 | 0.38 |
| P(body-shove ≤ 0.20 \| zone) | high | high |
| **state 64102 strict (retention)** | **True** | **False (lost)** |
| attribution margin (median) | **+0.168** | −0.096 |
| L/R contact rate | 0.59 / 0.34 | 0.36 / 0.37 |

Even on the state both arms certify (64201), CONTROL's contact is cleaner (attribution 0.79 vs 0.62). STRATIFIED gained
one new strict state (64113) but lost two (64102, 64203) — a net regression in coverage and in every mechanism endpoint.

## §11 classification → NEGATIVE
`coverage_down = True` (3 < 4) and every conditional contact-quality endpoint is lower for STRATIFIED, plus the loss of
the reproducible 64102 competence. Not NO_EFFECT (the drops are material and correlated), not MECHANISM_ONLY_POSITIVE
(mechanism metrics went **down**, not up). Reweighting the same single-actor/single-critic replay toward high-quality +
contrastive contact did not fix the contact-strategy gap — it degraded it.

## §12 presentation videos (`.../stratified_50k/videos/`, honest labels; the 100k 64102 video is NOT overwritten)

| file | arm/state | strict | attribution | sha256 (in `comparison_video_manifest.json`) |
|---|---|---|---|---|
| `stratified_certified_64201.gif` | STRATIFIED 64201 | YES | 0.62 | `d97550e6…` |
| `stratified_new_strict_64113.gif` | STRATIFIED 64113 (newly strict) | YES | 0.66 | `0c9e8e24…` |
| `control_matched_64201.gif` | CONTROL 64201 (matched) | YES | **0.79** | `32566b75…` |
| `control_retains_64102.gif` | CONTROL 64102 (STRATIFIED lost) | YES | 0.63 | `7733c46b…` |

## §13 decision → specify the 2-actor × 2-critic contact-mode experiment (NOT implemented)
`next_2actor_2critic_spec.md`: actors = {bilateral delivery, contact recovery}; critics = {task delivery `Q_task`,
mechanism validity `Q_mechanism`}, with the actor trading off task return against a mechanism-validity floor so zone
entry cannot be bought with a one-finger bulldoze. Delivery-v2b reward, strict predicate, K0 env unchanged. **Guardrail:**
harden the NEGATIVE with a ≥3-seed replication of this comparison before committing to the architectural build (a fresh
idea is not decisively dead from one seed).

## Commits (branch `exp/coin-contact-stratified-replay`)
1. `5e3d3db` add canonical contact-stratified replay sampling + 8 tests · 2. `91250bc` launch matched CONTROL/STRATIFIED
50k · 3. (this) report + comparison + videos + next-experiment spec.

## Provenance
Source commit `37d923a`; source checkpoint `sac_actor_best.pt`; seed 0; MuJoCo 3.10.0; device CPU; golden fingerprint
bit-identical (`1fe468b77`); 34 replay/SAC tests pass. `comparison.json` holds every per-state row.
