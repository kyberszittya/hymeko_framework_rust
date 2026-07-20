---
campaign: COIN competence-gated replay experiment (matched 12-seed UNIFORM vs GATED)
title: Competence-gated stratified replay — no advantage that survives the noise floor; strong-basin test could not run
date: 2026-07-20
branch: exp/coin-contact-stratified-replay
source_commit: 7fa3b91
verdict: NO_EFFECT (honest) — the pre-registered rule returns GATED_POSITIVE only VACUOUSLY (empty strong-basin group + CI spans zero + regression-to-mean confound)
---

# Competence-gated replay — matched 12-seed UNIFORM vs GATED

**Created-at:** 2026-07-20 23:40 JST. Only new variable: whether stratification is disabled online once competence is
established. GATED = contact-stratified replay while weak, then uniform after the gate fires; UNIFORM = ordinary replay
throughout. All other config fixed. Reuses the EXISTING competence gate (`comp`/`bc_coef`); the gate reads only the run's
own eval (no CONTROL leak), fires after 2 consecutive certified evals (hysteresis), and is irreversible.

## Competence-state → sampler mapping (documented, reused gate)
`bc_coef ∈ {1.0, 0.3}` (weak/progress) OR `first_strict` but `consec_strict < 2` → **STRATIFIED**;
`consec_strict ≥ 2` (established certified competence, confirmed at 2 consecutive evals; `bc_coef` 0.1/0.05) →
**UNIFORM**, irreversible. The gate fired **naturally** in every seed (switch steps 5000–32500).

## UNIFORM-reuse equivalence check → NOT equivalent → matched arm rerun
Re-running committed CONTROL seed 0 at the current commit gave **strict 2, not the committed 4** (eval curves differ).
The RL runs are **not bit-reproducible** (CPU BLAS-threading non-determinism; §3 permits this). Therefore the old 12
UNIFORM runs are **not byte-equivalent** and were **not reused**; a fresh matched UNIFORM arm was run alongside GATED
(24 runs, matched pairs per seed, same commit). **This same finding is the headline caveat below.**

## Per-seed (best checkpoint, 18 states) + gate switch

| seed | UNIF strict | GATED strict | switch | Δstrict |
|---|---|---|---|---|
| 0 | 2 | 2 | ev5/step12500 | 0 |
| 1 | 3 | 2 | ev2/step5000 | −1 |
| 2 | 3 | 2 | ev8/step20000 | −1 |
| 3 | 2 | 3 | ev5/step12500 | +1 |
| 4 | 2 | 3 | ev2/step5000 | +1 |
| 5 | 2 | 3 | ev9/step22500 | +1 |
| 6 | 2 | 2 | ev5/step12500 | 0 |
| 7 | 3 | 4 | ev2/step5000 | +1 |
| 8 | 2 | 2 | ev3/step7500 | 0 |
| 9 | 3 | 2 | ev4/step10000 | −1 |
| 10 | 1 | 3 | ev8/step20000 | +2 |
| 11 | 2 | 3 | ev13/step32500 | +1 |

Every GATED seed switched (own-run signal). **UNIFORM strict range this run: 1–3 — no seed reached ≥4.**

## Pooled 12-seed paired deltas (GATED − UNIFORM; bootstrap seed 20260720, B=10000)

| endpoint | mean | median | IQR | +/0/− | bootstrap 95% CI |
|---|---|---|---|---|---|
| strict count | +0.33 | +0.5 | 1.25 | 6/3/3 | **[−0.167, +0.833]** (spans 0) |
| certified coverage | +0.33 | +0.5 | 1.25 | 6/3/3 | [−0.167, +0.833] (spans 0) |
| P(attr ≥ 0.60 \| zone) | +0.004 | +0.017 | 0.22 | 6/1/5 | [−0.069, +0.076] (spans 0) |
| P(clean \| zone) | +0.018 | −0.014 | 0.25 | 5/1/6 | [−0.074, +0.112] (spans 0) |
| loose zone count | +0.33 | +0.5 | 1.25 | 6/3/3 | [−0.333, +1.0] (spans 0) |

## Basin coupling vs the prior −0.825
Spearman(UNIFORM strict, Δstrict) = **−0.378** (prior fixed-stratification −0.825). Weak group (UNIF ≤2, **n=8**) median
Δ **+1.0**. **Strong group (UNIF ≥4): n=0 — EMPTY.**

## Verdict: **NO_EFFECT** (the mechanical GATED_POSITIVE is vacuous)

The pre-registered decision script prints `GATED_POSITIVE`, but that is **not a trustworthy result**, for three
compounding reasons — reported plainly rather than banked as a win:

1. **The strong-basin test could not run (n=0).** The entire experiment hinges on *"does gating remove the strong-basin
   degradation?"* No UNIFORM run reached strict ≥4 this time (the prior experiment had 3/12 at ≥4 — pure run-to-run
   variance). The rule's "strong-group median ≥ 0" is satisfied only **vacuously**. The discriminating question is
   **unanswered**.
2. **The aggregate CI spans zero.** Δstrict median +0.5 but bootstrap 95% CI [−0.167, +0.833]; contact-quality deltas
   median ≈ 0. No endpoint is robustly non-zero.
3. **Regression-to-mean confound.** Δ = GATED − UNIFORM with a **noisy** UNIFORM (demonstrated ±2 strict run-to-run: the
   equivalence check gave 4→2 on the same config/seed). When UNIFORM is mostly low (1–3 this run), a second draw (GATED)
   mechanically reverts upward → a spurious positive Δ and a mechanically-weakened coupling (−0.378). The weak-group
   +1.0 is largely this artifact, not demonstrated gating benefit.

The gate itself is **correctly implemented and fires online as designed** (verified: 10 tests + natural switches in all
12 seeds). The *science* is inconclusive because the effect sizes (±0.5–1 strict) are **below the demonstrated
run-to-run noise floor (±2 strict)**, and the one discriminator that could cut through it (strong-basin seeds) did not
occur. This is the same lesson the 1→4→12-seed arc taught, now at the measurement layer: **the replay-sampler line is
noise-limited; single-draw-per-seed baselines make "basin dependence" partly a regression-to-mean artifact.**

## §11 next decision (NO_EFFECT branch) — SPEC ONLY
Retire fixed/gated replay-sampler interventions (they are below the resolution floor of this metric on this budget) and
**specify the minimal HyMeKo-native 2-actor × 2-critic contact-mode experiment** (`next_2actor_2critic_spec.md`, already
on record: actors = {bilateral delivery, contact recovery}; critics = {task delivery, mechanism validity}). Two
measurement guardrails to fold into ANY next experiment, learned here: (a) baselines must be **multi-run-per-seed** (or
seed-controlled) so a "basin" is a stable quantity, not a single noisy draw — otherwise paired deltas inherit a
regression-to-mean artifact; (b) rest every claim on the bootstrap CI, not the median sign.

## Commits (branch `exp/coin-contact-stratified-replay`)
1. `f12d9f8` add competence-gated selection + 10 tests · 2. (this) launch + 24-run matched data · 3. (this) report.

## Artifacts
24 run dirs `..._{uniform,gated}_50k_s{0..11}/` (each `run.json` = full 20-point curve + gate switch history).
`..._gated_50k_s0/gated_comparison.json` (per-seed rows, pooled stats, bootstrap, basin). `gated_compare.py`. Production
code unchanged except the gate hook (commit 1); golden bit-identical; 36 tests pass.
