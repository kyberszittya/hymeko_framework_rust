---
campaign: COIN clearance-transport curriculum (progressive SAC from the best GENERATOR checkpoint)
title: Curriculum extends certified delivery to short clear-start transport, not to presentation distance
date: 2026-07-21
branch: exp/coin-clearance-curriculum
source_commit: 18f54a9
verdict: SHORT_TRANSPORT_ONLY — certified clear-start delivery extended to clearance +0.0253 (base fails 0/10), below the +0.030 presentation grade
---

# Clearance-transport curriculum

**Created-at:** 2026-07-21 01:35 JST. ONE continuous SAC from the best GENERATOR checkpoint (`s2r0/actor_best.pt`,
sha `885bff21`), trained through 4 progressive footprint-clearance stages. Only variable = the generated training-config
distribution. Actor/critic/SAC/delivery-v2b reward/strict predicate/obs/action/BC anchor/replay sampler unchanged; no
hold shaping, no new reward, no n-step.

## Signed footprint clearance (§3)
`signed_clearance = disk_to_zone − coin_radius(0.02) − zone_half(0.04)`. Positive ⇒ coin footprint disjoint from the
target footprint. The generator's `move_to_clearance` moves the coin radially OUTSIDE the target into each band.

## Curriculum corpora (§5, frozen, `corpus_sha 2d5cf719`, train/held disjoint)
STAGE0 (+0.002–0.010) · STAGE1 (+0.010–0.030) · STAGE2 (+0.030–0.060) · STAGE3 (+0.060–0.120); 64 train + 24 held per
stage; left/right/symmetric leads; physical validity only (7 out-of-band rejects).

## Baseline (base GENERATOR checkpoint, BEFORE curriculum) on the curriculum held stages
| stage | coverage | loose | max cert clearance |
|---|---|---|---|
| STAGE0 | 2/24 | 13/24 | +0.006 |
| STAGE1 | 5/24 | 8/24 | +0.018 |
| STAGE2 | 0/24 | 0/24 | — (unreached) |
| STAGE3 | 0/24 | 0/24 | — (unreached) |
The base already does *some* short transport (+0.018); STAGE2/3 are entirely unreached.

## Result by stage (curriculum best checkpoint @ step 12500)
| stage | coverage | max cert clearance |
|---|---|---|
| STAGE0 | 2 | +0.006 |
| STAGE1 | **4** | **+0.0253** |
| STAGE2 | 0 | — |
| STAGE3 | 0 | — |
Advanced STAGE0→1→2 by step 10k (criterion). **STAGE2 (+0.030+) never reached** (loose 0 throughout). Training on the
unreachable STAGE2 caused **catastrophic forgetting** after step 12500 (max certified clearance collapsed to −9.9 in
later evals) — the 70/15/15 mix did not fully preserve competence. The best checkpoint is the early one (step 12500).

## §10 extension proof — the decisive test
On the strongest clear-start held state (**clearance +0.0253**, hash `04870b0e0357ecb5`, STAGE1), 10 deterministic runs
(identical initial-state hash each):

| policy | strict / 10 |
|---|---|
| **curriculum-trained** | **10/10** |
| base GENERATOR (starting) | **0/10** |
| zero-action control | **0/10** |

The base fails this state; the curriculum certifies it 10/10; footprints are disjoint (clearance > 0). This is a genuine
curriculum-acquired clear-start transport — but only **short** transport (+0.0253 < +0.030).

## §12 verdict: **SHORT_TRANSPORT_ONLY**
Reproducible certified delivery is acquired for clearance +0.002 to ~+0.025 (short transport, beyond the base's +0.018),
but NOT to the +0.030 presentation grade (§9). No held state with clearance ≥ +0.030 achieves ≥8/10 — STAGE2/3 remain
unreachable by this 1-actor × 1-critic policy on this budget.

## §11 artifacts (`.../videos/`, honest labels — NOT a presentation-grade headline)
| file | policy | strict | clearance | sha256 |
|---|---|---|---|---|
| `rl_short_transport_certified_delivery.mp4` | curriculum | YES | +0.0253 | `038da154…` |
| `rl_short_transport_certified_delivery.gif` | curriculum | YES | +0.0253 | `c3763326…` |
| `rl_short_transport_before_vs_after.mp4` | base FAIL → curriculum DELIVER (same hash `04870b0e`) | — | +0.0253 | `5ab04ea9…` |
Labeled "SHORT TRANSPORT — coin outside target, clearance +0.025 (< +0.030 presentation grade)". No
`rl_clear_start_certified_delivery.*` was produced — the presentation criterion (clearance ≥ +0.030, ≥8/10) is not met.

## §13 next decision (SHORT_TRANSPORT_ONLY)
**Retain this curriculum; next add `n_step=3` as the single isolated learning change** (the credit-assignment horizon is
the plausible limiter for pushing certified clearance from +0.025 to +0.030+). Also worth folding in: a stronger
anti-forgetting mix (raise the earlier-stage/orig-cert share) since STAGE2 training erased competence — but that is a
mix change, not a new learning mechanism. Do NOT build the 2-actor × 2-critic architecture yet.

## Provenance / commits (branch `exp/coin-clearance-curriculum`)
`b717546` curriculum generator+driver+corpora+run (source `18f54a9`; base ckpt sha `885bff21`); this report + videos.
64102 retention: not retained at the best checkpoint (False) — a real cost of the STAGE2 forgetting, consistent with the
short-transport-only verdict.
