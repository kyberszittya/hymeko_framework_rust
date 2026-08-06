---
title: Carry structured expert — the push→brake→release action-language is validated; it solves carry states the pi_0+offset class cannot
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
terminal: STRUCTURED_ACTION_CLASS_VALIDATED_CEM_DOMINATED_BY_RANDOM — best-in-class K6 0.833 vs offset 0.50 vs pi_0 0.125, +6 offset-unsolved states solved incl. transport (0→0.75); CEM dominated by random over the well-conditioned class
tags: [coin, carry, structured-primitive, push-brake-release, cem, existence-proof, phase4, action-language]
---

# CARRY_STRUCTURED_EXPERT_V1 — Phase 4: does a carry-specific action language beat the plateaued offset class?

The support-frontier proved the pi_0+offset candidate class plateaus at ~40% for contact_retention and does not scale with
support/search/budget — the CEM was searching in the wrong coordinate system. This tests a carry-SPECIFIC action language
(push→brake→release macro-action, ≈15 params, closed-loop phase transitions, frozen pi_0 only after a valid handoff) on a
FRESH held-out carry panel, against the plateaued offset class.

## Method
FRESH held-out strict-0 carry panel (seeds ≥9000, disjoint from the frontier dev panel; manifest-verified strict==0 ∧
gate_mult==1.0 ∧ family∈carry), 24 states (19 contact_retention, 4 transport, 1 braking). Four controllers on the same
states: PI_0; OLD_OFFSET_CEM (the plateaued class, 24×4, |offset|≤0.20); STRUCTURED_RANDOM (budget-matched random over the
structured params); STRUCTURED_CEM (24×4 over the macro-action). Lexicographic score: K6 ≻ handoff ≻ dwell ≻ −exit ≻ contact
≻ −effort ≻ −completion (handoff alone is not enough — the frontier showed handoff↑ K6 flat). Primary evidence =
**old-unsolved → structured-solved**, not aggregate coverage. No critic training.

## Result (24 fresh carry states)
| controller | K6 | handoff | any_exit |
|---|---|---|---|
| pi_0 | 0.125 | 0.125 | 0.125 |
| offset_CEM (plateaued class) | 0.500 | 0.500 | 0.042 |
| **structured_random** | **0.833** | 0.875 | 0.042 |
| structured_CEM | 0.708 | 0.792 | 0.083 |

**offset-UNSOLVED → structured-SOLVED: 6 states** ([6, 7, 9, 10, 15, 17]).
Per family: contact_retention pi_0 0.158 → offset 0.632 → **structured 0.737**; **transport pi_0 0 → offset 0 → structured
0.75** (a family the offset class NEVER solved); braking n=1 (0).

## Verdict — `STRUCTURED_ACTION_CLASS_VALIDATED_CEM_DOMINATED_BY_RANDOM`
The carry action-language is **validated**. Best-in-class K6 **0.833** (random) / 0.708 (CEM) ≫ offset **0.50** ≫ pi_0
0.125; it delivers **6 states the offset class provably fails**, including **transport (offset 0 → 0.75)**, with
full-containment exit not worse. Every class-validity condition is met: coverage ≫ 0.40, new states, K6 rises (not just
handoff), exit not worse, structured ≫ offset. This confirms the frontier's diagnosis: the limit was the *coordinate
system*, not the amount of search — a carry-fitted push→brake→release language reaches carry states that no support-bounded
perturbation of the settling-tuned pi_0 could.

**The one caught nuance:** structured_RANDOM (0.833) > structured_CEM (0.708). The class is so well-conditioned (≈15
physically-meaningful params) that random shooting over it beats the CEM — the CEM optimizer, not the class, is the weak
part. So the DAgger expert should be **structured-random (or a tuned optimizer, e.g. CMA-ES / more iters)**, not the current
CEM. This is a search-method finding, not a class-validity failure.

## Next lever (Phase 4b — no more mandatory intermediate audits)
Proceed to the DAgger carry actor, per the plan:
1. **Label** with the best structured expert (structured-random, or a tuned structured optimizer) — receding-horizon:
   re-solve per state, execute/label the first low-level 4D action, replan.
2. **BC-init a low-level continuous carry actor** (variant A: predicts the 4D action per state, not the macro-params — the
   structured CEM is the teacher, not the deployed architecture), then **DAgger** on-distribution corrections.
3. **update-0 eval** (the BC clone ≈ the expert's coverage), then **SAC/TD3** reward-driven refinement; **frozen settling
   pi_0 stays downstream** after the handoff.
Re-confirm coverage at larger n in passing (this is a first-pass 24-state fresh panel; the 0.833 is a data point, and
braking is absent / transport n=4).

## Honest limitations
- First-pass, 24 fresh states; contact_retention-primary (n=19), transport n=4 (but 0→0.75 is a strong exploratory signal),
  braking n=1. Re-confirm the coverage number on a larger fresh panel during Phase 4b.
- The macro-action is a fixed 3-phase (push/brake/release) structure with time+state-triggered transitions; a richer or
  learned phase structure could raise coverage further, but is not required to justify Phase 4b (the class already clears
  the gate).

## Files
- lib `hymeko_rl/coin_delivery/coin_carry_structured.py` — `structured_carry_rollout` (closed-loop push→brake→release →
  frozen pi_0), `structured_score` (lexicographic), `structured_cem`, `structured_random`.
- entry `experiments/…/rl_entry/coin_carry_structured_expert.py`; result `…/carry_structured_expert_v1.json`.
- test `test_structured_carry_primitive_and_cem` (keys, determinism, no-mutation, lexicographic K6-dominance). 27 tests
  pass, ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; FRESH held-out strict-0 carry panel seeds ≥9000
(disjoint from the frontier dev panel `d7f052ff` and train/dev banks). Structured CEM 24×4, seeds {200,300,400}+i
(deterministic); offset CEM at its best-known setting (|offset|≤0.20, len 30, 24×4). H=160. No training, no reward/task
change, no CORE.YAML items.
