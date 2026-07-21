---
campaign: COIN STAGE-2 n-step transport experiment (matched n_step=1 vs n_step=3)
title: Three-step returns do not crack the +0.030 transport barrier (no-effect)
date: 2026-07-21
branch: exp/coin-clearance-curriculum
source_commit: 0c2b908
verdict: NO_EFFECT — STAGE2 (+0.030+) unreached by both arms; every matched STAGE2 delta is exactly zero
---

# STAGE-2 n-step transport experiment

**Created-at:** 2026-07-21 01:50 JST. Isolated single-variable test of the SHORT_TRANSPORT_ONLY next-step hypothesis:
*does 3-step credit assignment let the existing 1-actor × 1-critic policy learn the +0.030 STAGE2 transport horizon?*
Matched CONTROL (n_step=1) vs TREATMENT (n_step=3), 8 pairs (4 seeds × 2 reps), thread-pinned, continued from the
curriculum best checkpoint (`run_s0/actor_best.pt`, sha `39551de3`), trained on the frozen STAGE2 corpus with the
committed 70/15/15 mix. Only variable: `n_step`.

## n-step implementation (canonical, in place — commit `9520e58`)
`ReplayBuffer.sample_nstep(n_step, gamma)` → `(obs, act, R, next_obs, done, disc)`, `R=Σ_k γ^k r_{t+k}`, `disc=γ^K`;
stops at termination, never crosses an episode boundary or the ring head (truncation → bootstrap from the last valid
next_obs). `SACConfig.n_step`; `train_sac` threads a **per-sample discount** into the critic target
`y = R + disc·(1−d)·q_next`. `n_step=1` is byte-identical to before. **8 regression tests pass** (γ-powers,
bootstrap-after-3, stop-at-termination, no-episode-crossing, 1-step==sample, demo/online identical, deterministic,
metadata-aligned).

## Per-pair (best checkpoint) — n_step=1 → n_step=3
| pair | STAGE2 cov | STAGE2 max clearance | STAGE1 retention | strong (+0.0253) |
|---|---|---|---|---|
| s0r0 | 0→0 | — → — | 9→3 | T→F |
| s0r1 | 0→0 | — → — | 2→5 | F→F |
| s1r0 | 0→0 | — → — | 5→6 | F→F |
| s1r1 | 0→0 | — → — | 2→5 | F→F |
| s2r0 | 0→0 | — → — | 4→7 | T→F |
| s2r1 | 0→0 | — → — | 4→7 | F→T |
| s3r0 | 0→0 | — → — | 1→8 | F→T |
| s3r1 | 0→0 | — → — | 6→7 | F→F |

## Pooled 8-pair paired deltas (n3 − n1; bootstrap seed 20260721, B=10000)
| endpoint | mean | median | +/0/− | bootstrap 95% CI |
|---|---|---|---|---|
| **STAGE2 certified coverage** | **0** | **0** | 0/8/0 | **[0.0, 0.0]** |
| STAGE2 loose entry | 0 | 0 | 0/8/0 | [0.0, 0.0] |
| STAGE2 max certified clearance | 0 | 0 | 0/8/0 | [0.0, 0.0] |
| STAGE1 retention coverage | +1.88 | +3 | 7/0/1 | [−0.625, +4.0] (spans 0) |
| strong (+0.0253) retention | 0 | 0 | 2/4/2 | [−0.5, +0.5] |
| 64102 retention | 0 | 0 | 1/6/1 | [−0.375, +0.375] |

## Verdict: **NO_EFFECT**
STAGE2 (+0.030–0.060) is **completely unreached by both arms** — coverage, loose entry, and max certified clearance are
**identically zero for all 8 pairs** (bootstrap CI [0, 0]); **no STAGE2 state is certified by either n_step=1 or
n_step=3.** The 3-step credit horizon did not crack the +0.030 barrier. Not NSTEP_CLEAR_START_POSITIVE (no STAGE2
certification), not NSTEP_PROGRESS_POSITIVE (STAGE2 loose/clearance deltas exactly 0), not NEGATIVE (retention if
anything leans *better* for n3).

The one non-zero signal is **STAGE1 retention** (n3 +1.88 mean, +3 median, 7/8 pairs positive) — 3-step returns *may*
stabilize the earlier competence against the STAGE2-training forgetting the curriculum suffered — but the CI spans zero,
so it is a lean, not a result, and it does nothing for the transport target. No §10 causal comparison (no qualifying
STAGE2 state); **no §11 clear-start demo** (the presentation criterion, STAGE2 clearance ≥ +0.030 reproducibly certified,
is not met).

## §13 next decision (NO_EFFECT) — SPEC ONLY
The generator + longer credit horizon are **insufficient** for long-range transport; the remaining lever is
representational. `next_factorial_spec.md`: the minimal HyMeKo-native **actor × critic factorial** — 1×1 / 2×1 / 1×2 /
2×2 (actors = {bilateral delivery, contact recovery}; critics = {task delivery `Q_task`, mechanism validity
`Q_mechanism`}) — same env/reward/predicate/corpora, matched thread-pinned multi-seed, judged on the STAGE2 certified
coverage bootstrap CI with retention guards. **Not implemented in this task.**

## Commits (branch `exp/coin-clearance-curriculum`)
`9520e58` n-step ReplayBuffer/SAC + 8 tests · `<driver>` matched experiment · `0b53c6c` 16-run data + analysis ·
(this) report + factorial spec. Golden bit-identical; 47 replay/SAC tests pass; production n-step path off by default
(`n_step=1`).
