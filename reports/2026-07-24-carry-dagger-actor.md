---
title: Carry DAgger actor — pipeline corrected (warm-started receding teacher); update-0 fails first-pass on a weak teacher + multimodal labels
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
terminal: UPDATE0_FAILED_FIRST_PASS_WEAK_TEACHER_AND_MULTIMODAL_LABELS — methodology correct, but the teacher budget is far below the validated 0.833 expert and DAgger MSE rises (label multimodality); NOT a distillation-fails verdict
tags: [coin, carry, dagger, bc, teacher, update-0, first-pass, multimodal-labels, feedback-admissible]
---

# CARRY_DAGGER_ACTOR_V1 — Phase 4b part 1: distil the carry expert; update-0 fails first-pass (correctly diagnosed)

Building the deployable carry actor from the validated structured expert. A mid-build methodological error was caught and
fixed before it could contaminate a result.

## Contract fix (caught before it mattered)
The first draft labeled BC with the **whole open-loop trajectory** of a strong s0-plan — reintroducing exactly the
quarantined open-loop-as-feedback error (a_t is optimized for s0, not for the actual s_t after student drift). Corrected:
- **PLAN_ONLY** — the full structured plan / warm start / provenance (`teacher_openloop_plan`, provenance
  `OPEN_LOOP_PLAN_ONLY`) — never fed as BC action labels.
- **FEEDBACK_ADMISSIBLE** — the first 4D action **replanned from the current state**. The teacher is a **warm-started
  receding-horizon** search: strong initial solve from s0, then cheap warm-started replans (centred on the previous plan)
  at each strict-0 step; every label is `first_action_of_theta(θ, current_state)`; ABSTAIN when no warm plan reaches a
  handoff. Cheap because warm-started, correct because every label is replanned.

Actor: low-level obs48→4D, acts only at strict-0 under the carry gate, hands off to the FROZEN pi_0 at strict≥1.

## Result (first-pass; TRAIN 20 / disjoint EVAL 20 held-out carry states)
- Teacher (warm-started receding): K6 **5/20**, abstained 12, **33 labels total**. BC MSE 1.25; **DAgger MSE 1.25 → 1.94
  → 2.61 (rising)**.
- Update-0 eval (held-out): pi_0 **0.15** | structured_expert **0.50** | BC **0.05** | DAgger **0.05**.

**Verdict: `UPDATE0_FAILED_FIRST_PASS_WEAK_TEACHER_AND_MULTIMODAL_LABELS`.** BC/DAgger (0.05) are below pi_0 (0.15). This is
NOT a distillation-fails verdict (no conclusions from a weak first-pass) — two concrete, fixable causes:
1. **Teacher too weak / sparse bank.** Teacher K6 5/20 and only 33 labels: the budget (48 strong / 8 warm, H100) is far
   below the validated 0.833 expert (96–192 shots × H160); even the expert reads only 0.50 here. The bank is tiny and
   easy-biased.
2. **Multimodal labels.** DAgger MSE *rises* (1.25→2.61): the structured search proposes *different* good first actions for
   similar states, so a deterministic MSE-BC averages the multimodal target into a poor mean — the actor regresses to a
   bad average, not a good mode.

## Next lever (needs your go — strengthen BEFORE concluding)
1. **Teacher at the validated budget** (96–192 shots, H160) so the bank is dense and drawn from consistently-good plans.
2. **Address label multimodality** (the rising DAgger MSE is the tell): re-solve each state with a FIXED warm-start / pick
   a canonical solution branch, filter inconsistent labels, or use a stochastic/mixture actor instead of deterministic
   MSE-BC.
3. **More train states + a relabel queue** (re-solve the `OPEN_LOOP_PLAN_ONLY` states individually as feedback labels).
Re-run update-0; only after BC/DAgger > pi_0 (physical K6, exit not worse) does SAC/TD3 from that checkpoint follow.

## Files
- lib `hymeko_rl/coin_delivery/coin_carry_dagger.py` (`teacher_warmstart_bank` — receding, replanned labels, abstain;
  `teacher_openloop_plan` PLAN_ONLY; `carry_actor_rollout`; `train_bc`; `make_carry_actor`); `coin_carry_structured.py`
  (`first_action_of_theta`, `structured_random_around` warm search, `structured_carry_rollout` capture).
- entry `experiments/…/rl_entry/coin_carry_dagger_actor.py`; result `…/carry_dagger_actor_v1.json`.
- test `test_structured_carry_primitive_and_cem` extended (first_action_of_theta phase logic, warm-started teacher
  determinism + no-mutation). 27 tests pass, ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; disjoint TRAIN (seeds 9000–10400) / EVAL (10500–12000)
held-out strict-0 carry panels, manifest-verified. Teacher 48-strong/8-warm shots H100, 2 DAgger iters; deterministic
seeds. First-pass, weak-teacher setup — not a policy verdict. No CORE.YAML items.
