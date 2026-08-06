---
campaign: COIN contact-stratified replay — matched multi-seed replication (seeds 0-3)
title: Does the seed-0 NEGATIVE hold? No — the effect is seed-sensitive (mean-reverting)
date: 2026-07-20
branch: exp/coin-contact-stratified-replay
source_commit: d56f692
verdict: SEED_SENSITIVE (4 seeds) — SHARPENED to BASIN_DEPENDENT by the 12-seed extension (2026-07-20-coin-contact-replay-12seed.md)
---

# Contact-stratified replay — matched multi-seed replication

**Created-at:** 2026-07-20 21:25 JST. Fixed experiment from `d56f692` (env / reward / strict predicate / actor+critic
architecture / BC anchor / replay implementation / state splits / eval states / 50k budget / checkpoint ranking all
unchanged; only difference = CONTROL uniform vs STRATIFIED contact-stratified replay). Seeds 1–3 added to the committed
seed 0. Within each seed, CONTROL and STRATIFIED share the network-init seed, exploration policy, source checkpoint
(`sac_actor_best.pt`), demo corpus, state splits and eval order; **eval states are the fixed explicit 18 (4 DEMO + 14
VAL), never seed-selected.** Six new runs executed as matched pairs (2-concurrent, mirroring the seed-0 resource
profile), seeds sequential.

## Verdict: **SEED_SENSITIVE** — the seed-0 NEGATIVE does not replicate

The best-checkpoint paired deltas (STRATIFIED − CONTROL) on the 18-state eval set:

| seed | CONTROL strict / cov | STRAT strict / cov | Δstrict | Δcov | ΔP(clean\|zone) | ΔP(attr\|zone) | STRAT 64102 |
|---|---|---|---|---|---|---|---|
| 0 | 4 / 4 | 3 / 3 | **−1** | −1 | −0.07 | −0.17 | False |
| 1 | 2 / 2 | 4 / 4 | **+2** | +2 | +0.18 | +0.18 | **True** |
| 2 | 2 / 2 | 3 / 3 | **+1** | +1 | 0.00 | +0.38 | False |
| 3 | 4 / 4 | 2 / 2 | **−2** | −2 | −0.24 | +0.10 | False |

Aggregate paired deltas across seeds 0–3:

| endpoint | values | median | IQR | +/0/− |
|---|---|---|---|---|
| strict count | [−1, +2, +1, −2] | **0** | 2.5 | 2/0/2 |
| certified-state coverage | [−1, +2, +1, −2] | **0** | 2.5 | 2/0/2 |
| P(clean mechanism \| zone) | [−0.07, +0.18, 0.00, −0.24] | −0.03 | 0.16 | 1/1/2 |
| P(attr ≥ 0.60 \| zone) | [−0.17, +0.18, +0.38, +0.10] | +0.14 | 0.20 | 3/0/1 |
| loose zone-entry count | [−1, +1, 0, +2] | +0.5 | 1.5 | 2/1/1 |

The primary endpoints (strict count, coverage) have **median 0 with IQR 2.5 spanning −2 to +2** — 2 seeds positive, 2
negative. The direction changes materially across seeds ⇒ **SEED_SENSITIVE** (not CONFIRMED_NEGATIVE, not NO_EFFECT, not
REVISED_POSITIVE). State 64102 is retained by STRATIFIED in only 1/4 seeds.

## The identified single sensitivity (per the SEED_SENSITIVE decision rule)

The sign of the stratified effect is **determined by the CONTROL arm's outcome on that seed**:

| CONTROL strict | seeds | STRATIFIED Δstrict |
|---|---|---|
| **weak (2)** | 1, 2 | **+2, +1** (helps) |
| **strong (4)** | 0, 3 | **−1, −2** (hurts) |

Perfect rank separation: contact-stratified replay **lifts the seeds where uniform replay landed in a poor basin and
drags down the seeds where uniform already found a strong basin.** It acts as a **variance-reducing / mean-reverting
regularizer on the certified-delivery outcome**, not a consistent directional improvement. This — the dependence of the
effect sign on the uniform baseline's basin — is the single configuration sensitivity to control before any architecture
change. P(attr|zone) is the one endpoint with a mild consistent tilt (+3/−1, median +0.14), so the sampler does nudge
fingertip attribution up on average, but not enough to move strict delivery consistently.

## Consequence for the earlier conclusions
- The seed-0 report's **NEGATIVE is overturned** — it was one draw from a high-variance distribution whose median effect
  is ~0. Corrected there with a pointer to this replication.
- The §13 "specify the 2-actor × 2-critic" recommendation was **premature**: it was gated on CONFIRMED_NEGATIVE/NO_EFFECT,
  and the replication returned neither. Do **not** start that architecture on this evidence.

## Next decision (SEED_SENSITIVE branch)
Inspect the paired seed differences (done: the effect sign tracks the CONTROL basin) and **control that sensitivity
before changing architecture**. Concretely, the next single-variable step is to remove the basin dependence — e.g.
fix the initial-basin variance (more/again seeds, or a shared critic warm-start) and re-measure whether the +P(attr|zone)
tilt converts to strict delivery once the mean-reversion confound is removed. NOT an architecture change; NOT more
ratio tuning of the sort already shown to be seed-dominated. Architecture (2-actor × 2-critic) remains unbuilt.

## Artifacts
Per-seed run dirs `experiments/2026_07_20_coin_contact_replay_{control,stratified}_50k[_s1..s3]/` (each `run.json` holds
the full 20-point eval curve + best-checkpoint metrics). Aggregate `..._stratified_50k/multiseed_comparison.json`
(per-seed rows + paired deltas + sign counts). Analysis `multiseed_compare.py`. Golden fingerprint bit-identical; 34
replay/SAC tests pass; production code unchanged since `5e3d3db` (no runtime defect appeared).
