---
campaign: COIN contact-stratified replay — 12-seed extended matched replication (seeds 0-11)
title: Fixed contact-stratified replay has no average effect — it is basin-dependent (mean-reverting)
date: 2026-07-20
branch: exp/coin-contact-stratified-replay
source_commit: 78405e4
verdict: BASIN_DEPENDENT — average ≈ 0, but the effect sign inversely tracks the CONTROL basin (Spearman -0.825)
---

# Contact-stratified replay — 12-seed extended replication

**Created-at:** 2026-07-20 22:45 JST. Fixed experiment from `78405e4` (env / reward / strict predicate / actor+critic
architecture / initial checkpoint / BC gate / replay algorithms / demo corpus / state splits / 18 eval states / 50k
budget / checkpoint ranking all unchanged; only difference = uniform vs contact-stratified replay). Seeds 4–11 added to
seeds 0–3. Matched pairs share init + stochastic-seed policy; the sampler is the only treatment difference; eval states
are the fixed explicit 18 (never seed-selected). 16 new runs; production code unchanged (no runtime defect).

## Verdict: **BASIN_DEPENDENT**

The average treatment effect is ≈ 0 with a bootstrap CI spanning zero, but the per-seed effect **inversely tracks the
CONTROL arm's competence** (Spearman −0.825): weak basins improve, strong basins degrade.

## Per-seed paired deltas (STRATIFIED − CONTROL, best checkpoint on 18 states)

| seed | CTRL strict | STRAT strict | Δstrict | Δcov | ΔP(attr\|z) | ΔP(clean\|z) |
|---|---|---|---|---|---|---|
| 0 | 4 | 3 | −1 | −1 | −0.17 | −0.07 |
| 1 | 2 | 4 | +2 | +2 | +0.18 | +0.18 |
| 2 | 2 | 3 | +1 | +1 | +0.38 | 0.00 |
| 3 | 4 | 2 | −2 | −2 | +0.10 | −0.24 |
| 4 | 3 | 3 | 0 | 0 | +0.21 | +0.02 |
| 5 | 4 | 3 | −1 | −1 | −0.05 | −0.07 |
| 6 | 3 | 2 | −1 | −1 | −0.18 | −0.04 |
| 7 | 3 | 3 | 0 | 0 | −0.25 | −0.12 |
| 8 | 3 | 2 | −1 | −1 | −0.12 | −0.25 |
| 9 | 3 | 3 | 0 | 0 | 0.00 | +0.11 |
| 10 | 2 | 3 | +1 | +1 | −0.12 | +0.25 |
| 11 | 3 | 2 | −1 | −1 | +0.09 | −0.09 |

## Pooled 12-seed statistics (bootstrap seed 20260720, B = 10000)

| endpoint | mean | median | IQR | min/max | +/0/− | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| strict count | −0.25 | −0.5 | 1.25 | −2 / +2 | 3/3/6 | **[−0.83, +0.42]** (spans 0) |
| certified coverage | −0.25 | −0.5 | 1.25 | −2 / +2 | 3/3/6 | [−0.83, +0.42] (spans 0) |
| P(attr ≥ 0.60 \| zone) | +0.004 | −0.027 | 0.25 | −0.25 / +0.38 | 5/1/6 | [−0.094, +0.108] (spans 0) |
| P(clean \| zone) | −0.027 | −0.056 | 0.14 | −0.25 / +0.25 | 4/1/7 | [−0.106, +0.055] (spans 0) |
| loose zone count | +0.25 | 0 | 1.25 | −1 / +2 | 4/5/3 | [−0.33, +0.83] (spans 0) |

Every endpoint's bootstrap CI **comfortably spans zero** (descriptive, not a significance claim from 12 seeds). Strict
and coverage lean mildly negative (median −0.5) but with 3 positive seeds. There is **no consistent contact-quality
improvement** to compensate (P(attr) median −0.03, P(clean) median −0.06). → not AVERAGE_POSITIVE.

## Basin interaction (the decisive structure)

| group | seeds (n) | median Δstrict |
|---|---|---|
| **weak CONTROL (strict ≤ 2)** | 1, 2, 10 (3) | **+1.0** (stratification helps) |
| **strong CONTROL (strict ≥ 4)** | 0, 3, 5 (3) | **−1.0** (stratification hurts) |

**Spearman(CONTROL strict, Δstrict) = −0.825.** The treatment delta is strongly, inversely coupled to the uniform
baseline's basin. Contact-stratified replay is a **mean-reverting regularizer on certified delivery**: it pulls weak
runs up toward, and strong runs down toward, a common ~3/18 mean. This is why the seed-0 NEGATIVE and the seed-1
POSITIVE were both real — they were the strong-basin and weak-basin tails of the same basin-conditioned effect.

## §-decision → BASIN_DEPENDENT
`average ≈ 0` (CI spans zero, no compensating contact-quality gain) **and** a clear inverse basin relationship. Not
AVERAGE_NEGATIVE (median only −0.5, 3 seeds positive, CI spans zero), not NO_AVERAGE_EFFECT (the inverse structure is
strong, not "no direction"), not AVERAGE_POSITIVE.

## Next experiment (BASIN_DEPENDENT branch) — SPEC ONLY, not implemented
Do **not** build N-actor × k-critic. Replace **always-on** stratification with **one competence-gated sampler
experiment** (`next_competence_gated_sampler_spec.md`): stratify the replay only while contact competence is weak, and
revert to uniform replay once certified/contact competence is established — reusing the *existing* competence gate
(`comp["progress_ok"]`/`first_strict`) that already drives `bc_coef`. This targets exactly the measured structure: apply
the regularizer where it helps (weak basins) and remove it where it hurts (strong basins). No architecture change; no
further always-on ratio tuning.

## Artifacts
24 run dirs `experiments/2026_07_20_coin_contact_replay_{control,stratified}_50k[_s1..s11]/` (each `run.json` = full
20-point eval curve + best-checkpoint metrics). `..._stratified_50k/twelveseed_comparison.json` (per-seed rows, pooled
stats, bootstrap, basin interaction). `multiseed12.py`. Golden bit-identical; 34 replay/SAC tests pass; production code
unchanged since `5e3d3db`.
