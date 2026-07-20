---
campaign: COIN F11 vs F12 semantic-critic contrast (1 actor × task critic vs 1 actor × task + Q_mechanism critic)
title: A separate semantic Q_mechanism critic does not crack the +0.030 transport barrier (no-effect)
date: 2026-07-21
branch: exp/coin-actor-critic-factorial
source_commit: e1e0961
verdict: NO_EFFECT — STAGE-2 (+0.030+) unreached by both cells (all 8 paired Δ=0, boot95 [0,0]); the mechanism-validity metrics the critic was designed to move span zero
---

# F11 vs F12 — does a second, *semantic* critic unlock long-range transport?

**Created-at:** 2026-07-21 06:05 JST. **ETA basis (met):** 16 runs × 50k steps, 8-parallel thread-pinned ≈ 8 min
wall (measured 480 s) + analysis/figure/GIF ≈ 15 min. The plan-first four-artifact bundle was **waived by the user's
explicit "EXECUTE NOW, do not ask for permission again"** directive for this bounded, fully-specified slice; this
report is the unit of acceptance.

## Question (isolated single variable)
The generator (GENERATOR_POSITIVE), clearance curriculum (SHORT_TRANSPORT_ONLY → +0.0253), and 3-step credit horizon
(NO_EFFECT) all left the **+0.030 STAGE-2 transport barrier** standing. The factorial spec named the remaining lever
as *representational*: a **second, semantic critic** — a mechanism-validity value `Q_mechanism` that (hypothesis)
"stops the policy trading clean bilateral contact for reach." This experiment runs the minimal decisive slice of that
factorial — **F11** (1 actor × task critic `Q_task`) vs **F12** (1 actor × task critic + a *separate* semantic
`Q_mechanism` critic) — everything else held identical, and asks whether the mechanism critic unlocks STAGE-2.

## What F12 adds (and what it deliberately does not)
`Q_mechanism` is **not** the twin-Q anti-overestimation pair. It is an independent value estimator (own critic
ensemble, own polyak target, own optimizer, **fresh Xavier-init — never inheriting task-critic weights**) that learns
the bounded per-step **mechanism-validity** target `r_mech(s') = both_contact ∧ ¬arm_body_contact` from canonical
**named** observation fields only (indices 28/29 via `field_index`), and contributes a fixed, **pre-registered**
`mech_coef = 0.5 · min_i Q_mech(s, a_π)` to the actor objective. It is encapsulated in one `_MechanismCritic` object
(present = F12, `None` = F11) — a structural variant as a class, not scattered `forward`-time flags (§6.5 #8). The task
SAC critic, the env reward, and the strict delivery predicate are **unchanged**; the two-actor bank and variable coin
sizes were **not** implemented (out of scope by directive). F11 is **byte-identical** to the pre-existing trainer
(default `critic_mode="TASK_ONLY"`; zero RNG consumed when the mechanism object is absent).

## Design
Matched, thread-pinned (`OMP/MKL/OPENBLAS=1`), 8 pairs = seeds {0..3} × reps {0,1}; 50k steps each; eval every 2500;
`n_step=1`; both cells continued from the **same** curriculum checkpoint (`run_s0/actor_best.pt`, sha `39551de3`),
trained on the frozen STAGE-2 corpus with the committed 70/15/15 mix. **Fresh matched F11 controls were re-run** — no
previous stochastic F11 numbers were reused (§9). Only variable: `critic_mode`.

## Per-pair best checkpoint (F11 → F12)
| pair | STAGE2 cov | STAGE1 retention | ΔS1 clean | ΔS1 bilat | ΔS1 attr |
|---|---|---|---|---|---|
| s0r0 | 0→0 | 6→3 (−3) | −0.39 | −0.25 | −0.03 |
| s0r1 | 0→0 | 5→7 (+2) | −0.11 | −0.01 | +0.38 |
| s1r0 | 0→0 | 6→5 (−1) | −0.08 | −0.08 | −0.30 |
| s1r1 | 0→0 | 5→5 (0) | +0.17 | +0.14 | −0.16 |
| s2r0 | 0→0 | 5→9 (+4) | −0.10 | +0.40 | +0.00 |
| s2r1 | 0→0 | 7→9 (+2) | +0.04 | +0.11 | −0.07 |
| s3r0 | 0→0 | 8→8 (0) | −0.10 | +0.25 | −0.20 |
| s3r1 | 0→0 | 7→9 (+2) | −0.01 | +0.07 | +0.15 |

## Pooled 8-pair paired deltas (F12 − F11; bootstrap seed 20260721, B=10000)
| endpoint | mean | median | +/0/− | bootstrap 95% CI |
|---|---|---|---|---|
| **STAGE2 certified coverage** (primary) | **0** | **0** | 0/8/0 | **[0.0, 0.0]** |
| STAGE2 loose entry | 0 | 0 | 0/8/0 | [0.0, 0.0] |
| STAGE2 max certified clearance | 0 | 0 | 0/8/0 | [0.0, 0.0] |
| STAGE1 retention coverage | +0.75 | +1 | 4/2/2 | [−0.625, +2.125] (spans 0) |
| strong (+0.0253) retention | +0.25 | 0 | 3/4/1 | [−0.25, +0.625] (spans 0) |
| 64102 retention | 0 | 0 | 1/6/1 | [−0.375, +0.375] (spans 0) |
| STAGE1 **P_bilat** (mechanism) | +0.078 | +0.089 | 5/0/3 | [−0.05, +0.209] (spans 0) |
| STAGE1 **P_clean** (mechanism) | −0.072 | −0.089 | 2/0/6 | [−0.186, +0.027] (spans 0) |
| STAGE1 **P_attr** (mechanism) | −0.028 | −0.049 | 2/1/5 | [−0.158, +0.121] (spans 0) |

Figure: [reports/figures/2026-07-21-f11-f12/paired_deltas.png](figures/2026-07-21-f11-f12/paired_deltas.png) — forest
plot (marker = median, bar = bootstrap 95% CI, zero line). Every CI touches zero.
Animated: [reports/figures/2026-07-21-f11-f12/f11_vs_f12_64102.gif](figures/2026-07-21-f11-f12/f11_vs_f12_64102.gif) —
matched F11 vs F12 (seed s2r0, the largest F12 retention gain) on the near state 64102: **both enter the zone
(loose=PASS), neither certifies (strict=FAIL)**; F12 is marginally cleaner *here* (attr 0.846 vs 0.773, dwell 48 vs 41)
but the pooled attribution/clean CIs span zero.

## Verdict: **NO_EFFECT**
- **Primary endpoint dead flat.** STAGE-2 (+0.030–0.060) is **completely unreached by both cells** — certified
  coverage, loose entry, and max certified clearance are **identically zero across all 8 pairs** (boot95 [0,0]); **no
  F12 run certifies any STAGE-2 state.** The separate semantic critic did **not** crack the +0.030 barrier. Not
  CRITIC_POSITIVE.
- **The mechanism critic did not even move the metrics it was designed to move.** Its stated job was to raise clean
  bilateral contact; the pooled STAGE-1 mechanism-validity deltas all span zero, and the sign is *inconsistent* —
  P_bilat leans slightly **positive** (+0.078) while P_clean leans slightly **negative** (−0.072) and P_attr negative
  (−0.028). Not CRITIC_MECHANISM_POSITIVE.
- **No degradation.** Retention guards (STAGE-1, strong-state, 64102) all span zero; STAGE-1 retention if anything
  leans mildly better (+0.75 mean, 4/8 positive) as it did for n-step, but the CI spans zero — a lean, not a result.
  Not CRITIC_NEGATIVE.
- **Not BLOCKED** — 16/16 runs completed rc=0.

The mechanism critic trained healthily throughout (loss finite `≈0.001–0.04`, `Q_mech` a distinct value function from
`Q_task` — verified live in every run log), so this is a genuine null of a *working* mechanism, not a plumbing
failure. No §10 causal comparison and no §11 clear-start demo (no qualifying STAGE-2 / clear-start success exists to
reproduce or present) — same as the n-step result on the same barrier.

## Interpretation (measured vs inferred, scoped)
**Measured:** with a fixed pre-registered `mech_coef=0.5`, a fresh-init semantic `Q_mechanism` added to the single
actor's objective, continued from the curriculum checkpoint on the frozen STAGE-2 corpus, produced **no STAGE-2
certification and no significant shift in mechanism-validity** across 8 matched seeds. **Inferred:** this is the fourth
distinct local intervention (contact-stratified replay, competence gate, n-step credit horizon, semantic critic) to
cap at the identical +0.030 wall — consistent with the barrier being **representational / capacity-structural**, not a
credit-assignment or contact-shaping deficit a value head on one actor can fix. **Still hypothesis (not closed):** a
single unswept coefficient and a single mechanism-target definition were tested; a coefficient sweep, an alternate
mechanism target, or the *structural* arm of the factorial (a second, dedicated **transport/recovery actor** — the B/D
cells, deliberately out of scope here) remain untested and could still move STAGE-2. This is one clean data point that
the *critic* axis alone does not, not a verdict that the factorial is dead.

## §13 next decision (NO_EFFECT) — SPEC ONLY, not implemented
The critic axis is exhausted for the single actor; the remaining untested factorial lever is the **structural actor
axis** (B: 2 actors × 1 critic; D: 2 actors × 2 critics), where a dedicated transport/recovery actor — not a value
head — supplies the missing long-range primitive the far-start geometry analysis showed no scripted primitive
provides. That is a distinct-class multi-actor build (`SAC_CONTACT_ACTOR_BANK`, currently validated-unsupported /
fail-loud), gated on the same STAGE-2 bootstrap-CI endpoint with the retention guards, and on the arm-repositioning
generator refinement (`next_geometry_farther_start_spec.md`) so a far start is geometrically actionable. Treat the
first structural result, when run, as a data point needing 8–12 matched seeds — not a verdict.

## Files touched
- `hymeko_rl/train/rl_config.py` (NEW, +84) — configurable policy/strategy/critic selection + loud validation + `mechanism_reward`.
- `hymeko_rl/train/sac.py` (+~70/−15) — `SACConfig.critic_mode`/`mech_coef`; `_MechanismCritic` class; F12 wiring; separate task/mech/BC logging; `diag_out`.
- `hymeko_rl/eval/paired_stats.py` (NEW, +42) — canonical percentile-bootstrap primitive (`boot_ci`, `paired_stats`).
- `hymeko_rl/viz/campaign_viz.py` (+45) — `plot_paired_deltas` forest plot (reusable matched-pair viz).
- `hymeko_rl/experiments/coin_nstep_exp.py` (+~10) — `critic_mode` axis threaded into the single-variable driver (default TASK_ONLY = n-step byte-identical).
- `hymeko_rl/experiments/coin_f11_f12_campaign.py` (NEW, +150) — 16-cell matched campaign launcher + §12 paired analysis + §13 classification.
- Tests (NEW): `test_sac_mechanism_critic.py` (11), `test_paired_stats.py` (6), `test_coin_f11_f12_campaign.py` (8), `test_campaign_viz.py` (+2).
- Data: `experiments/2026_07_21_coin_f11_f12/` (16 run dirs + manifest + comparison JSON); figures under `reports/figures/2026-07-21-f11-f12/`.

## CORE.YAML items touched
None. No dependencies added.

## Test results
| suite | count | result |
|---|---|---|
| `test_sac_mechanism_critic` (F11/F12) | 11 | pass |
| `test_paired_stats` | 6 | pass |
| `test_coin_f11_f12_campaign` | 8 | pass |
| `test_campaign_viz` | 5 (+2 new) | pass |
| SAC/replay regression (`test_sac`, `test_nstep_replay`, `test_replay`, competence-gate, compiled-update, contact-stratified) | 52 | pass |

F11 byte-identical to the pre-existing trainer (regression test asserts default == explicit TASK_ONLY, identical actor
weights). `cargo`-free Python change; `ruff check` clean on all touched files; `mypy` clean on the new modules
(pre-existing mujoco-stub / reward.py errors are unrelated). §6.2: `train_sac` cyclomatic 44→48 (naive inline was 54;
class extraction recovered 6) — a **pre-existing** hard-ceiling waiver on a long-accreted training loop, my delta
minimized and all new `_MechanismCritic` methods within budget.

## Performance
| axis | value | budget |
|---|---|---|
| training wall / run (50k) | F11 197 s, F12 282 s (8-parallel, thread-pinned) | — |
| campaign wall (16 runs) | 480 s | — |
| peak RSS / run | **0.42 GB** | 16 GB cap ✓ (≈3.4 GB at 8-parallel peak) |
| F12 per-step cost | ~1.4× F11 (the extra critic forward+update) | — |

## Experiment provenance
- Git SHA `e1e0961` (working tree adds the uncommitted `experiments/2026_07_21_coin_f11_f12/` data + figures + this report).
- Host: Apple M5 Pro, 18 cores, 48 GB RAM, macOS (Darwin 25.5). torch 2.12.0, mujoco 3.10.0, numpy 2.4.6.
- Seeds: run_seed = seed·100 + rep, seeds {0..3} × reps {0,1}. Bootstrap seed 20260721, B=10000.
- Source checkpoint sha `39551de3`; STAGE-2 corpus + 70/15/15 mix frozen (curriculum artifacts).
- RL is not bit-reproducible (CPU BLAS threading; §3 carve-out) → the verdict rests on the 8-pair matched bootstrap CI, not single-run reproduction.

## Open issues / follow-ups
- The `bc_coef`/competence-gate is **configured but inert** in this continuation driver (no demo stream threaded) — BC contribution is `nan` for **both** cells; the contrast stays clean (matched), but a future demo-anchored variant would exercise the BC term the mechanism split logs.
- Only `mech_coef=0.5` and one mechanism-target definition tested; a sweep is the natural robustness follow-up before any strong claim about the critic axis.
- The 20+ scattered ad-hoc bootstrap snippets (`grep bootstrap hymeko_rl/experiments`) should migrate to `paired_stats.boot_ci` (this task seeds the canonical home; migration is a separate cleanup).
