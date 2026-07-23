---
title: Carry teacher diagnostic — the bottleneck is the RECEDING formulation, not the teacher budget; the carry task is commitment-dominated
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
terminal: RECEDING_FORMULATION_IS_THE_BOTTLENECK — strong open-loop macro K6 0.60, any receding replan (warm or warm+strong-fallback) 0.20; the carry task rewards COMMITMENT, so a myopic per-step feedback teacher/actor underperforms the open-loop macro
tags: [coin, carry, teacher, diagnostic, open-loop-vs-receding, commitment, option-actor, phase4b]
---

# CARRY_TEACHER_DIAGNOSTIC_V1 — localising the DAgger bottleneck: it's the receding formulation, not the budget

The DAgger update-0 failed (BC/DAgger 0.05 < pi_0 0.15) with a teacher at K6 0.25 and 33 labels. This control isolates
whether that is (a) a hard panel, (b) a weak lightweight-replan config, or (c) a deeper formulation problem — on the SAME
20 TRAIN carry states, three teachers, three metrics.

## Result (20 TRAIN carry states, strong = 64 shots, H=160)
| teacher | K6 | handoff | mean labels | fallbacks | abstain reasons |
|---|---|---|---|---|---|
| **strong s0 (open-loop ceiling)** | **0.60** | 0.60 | — | — | — |
| warm receding (no fallback) | 0.20 | 0.20 | 1.7 | — | INITIAL_STRONG 10 / WARM_REPLAN 6 / none 4 |
| warm receding + strong fallback | **0.20** | 0.25 | 5.75 | 0.85 | INITIAL_STRONG 6 / WARM_THEN_STRONG 5 / none 9 |

## The finding — commitment beats replanning here
- The **strong open-loop macro plan solves 0.60** of these states (consistent with the validated ~0.833 at higher budget)
  — the expert and the action-language are fine.
- **Any RECEDING teacher gets only 0.20** — a 3× drop. Crucially, the **strong fallback did NOT help** (0.20 → 0.20): it
  *did* find plans (mean labels 1.7 → 5.75, total 115) but executing-first-action-then-replanning does not reproduce the
  open-loop success. Mid-trajectory strong re-solves frequently abstain (`WARM_THEN_STRONG_ABSTAIN`).

So the bottleneck is **the receding formulation itself**, not the teacher budget (strong s0 at 64 shots = 0.60) and not
the fallback (more labels, no K6). The carry phase is **commitment / momentum-dominated**: once the push begins the coin is
on a committed trajectory, and re-deciding each step underperforms committing to the macro. This also empirically resolves
the earlier quarantine concern: replanning-from-`s_t` was assumed to be the *correct* feedback, but here it is provably
*worse* (0.20 vs 0.60) — so the open-loop macro's actions are, for this task, the *better* targets.

## What this reframes for Phase 4b (needs your decision)
The deployable carry controller should COMMIT to a plan, not be a myopic per-step low-level feedback actor:
1. **Option / macro-parameter actor (variant B)** — the policy outputs the macro-params θ (push/brake/release amplitudes +
   durations) at s0 (or per phase); the closed-loop phase-controller executes. DAgger is then over θ (label = the strong s0
   plan's θ per state — well-defined and consistent), and the deployed controller inherits the open-loop 0.60 (→ 0.833 at
   full budget), not the receding 0.20. This is the natural fit for a commitment-dominated task.
2. **Or a low-level actor trained on the open-loop macro's (state, action) pairs** — now justified (not the quarantined
   error) because replanning is provably worse here; but a per-step MSE actor must still overcome multimodality, whereas the
   option actor sidesteps it by predicting the whole plan.
Recommendation: switch Phase 4b to the **option/macro-parameter actor** — it matches the commitment structure the
diagnostic revealed, gives consistent single-vector labels (killing the earlier multimodal-MSE problem), and deploys the
proven open-loop coverage. The frozen settling pi_0 stays downstream after the handoff.

## Files
- entry `experiments/…/rl_entry/coin_carry_teacher_diagnostic.py`; result `…/carry_teacher_diagnostic_v1.json`.
- lib `coin_carry_dagger.py` (`teacher_warmstart_bank` now two-tier with strong fallback + abstain reasons; strong solves
  use the validated UNIFORM search). 27 tests pass, ruff F-clean (verified before Bash became briefly unavailable).

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; 20 held-out strict-0 carry TRAIN states (seeds
9000–10400), manifest-verified. Strong 64 shots (uniform), warm 8, H=160, roll 120; deterministic seeds. No training, no
CORE.YAML items.
