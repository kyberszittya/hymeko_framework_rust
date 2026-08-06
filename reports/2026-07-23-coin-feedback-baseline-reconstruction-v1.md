---
title: Coin feedback baseline reconstruction V1 — gold controllers + H=30 teacher qualification
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: H30_TEACHER_UNQUALIFIED — LOSES_REQUIRED_CONTACT, INCREASES_TARGET_EXIT
tags: [coin, reconstruction, gold-baseline, teacher-qualification, receding-horizon, phase-conditional-contact, no-training]
---

# COIN_FEEDBACK_BASELINE_RECONSTRUCTION_V1 — the H=30 teacher is a strong-but-unqualified improver

No training. Local residual / transactional-TD3 / chunk-policy experiments are stopped. This returns to a clean,
correctly-qualified baseline: it freezes the completed arc, reconstructs four gold controllers in one canonical harness
with verified identities, and runs the **load-bearing full-horizon H=30 teacher qualification** on 31 disjoint dev
late-start states. The verdict selects the single next command.

## 1 — Frozen arc (historical evidence, artifacts unmodified)

All eight accepted outcomes were located on disk and recorded read-only in `baseline_reconstruction_v1.json:frozen_arc`
(verdict token → artifact path → sha256-16). 8/8 exist and were not modified:

`PHASE_GATED_RESIDUAL_CRITIC_ROUTE_BLOCKED`, `HOLD_SIGNAL_DOMINATED_BY_HARM`, `PHASE_SWITCHED_TD3_STAGE1_NO_IMPROVEMENT`,
`PHASE_SWITCHED_TD3_STAGE1B_NO_IMPROVEMENT`, `PHASE_SWITCHED_TD3_STAGE1C_NO_IMPROVEMENT`,
`TRANSPORT_DWELL_TD3_NO_IMPROVEMENT`, `FEEDBACK_CHUNK_WARMSTART_V2_STILL_UNDERPERFORMS`, `CHUNK_SUPERVISED_M1_FEEDBACK_NO_GAIN`.

## 2 — Verified controller / checkpoint identities

| item | identity | verified |
|---|---|---|
| B — frozen pi_0 | `pi0_shared_clip_actor.pt` sha256 `1902454ca7a7…` | ✓ matches manifest prefix `1902454c` |
| A — recovered E-approach | `E_valselect_v2.pt` sha256 `7dbbf1a7782f…` | ✓ matches manifest prefix `7dbbf1a7` |
| C/D — H=30 planner config | horizon 30, pop 40, iters 6, elite 8 | ✓ accepted-canonical (pilot SHA `1115ade3…`) |
| reward | `data/robotics/galambos_task_deliver_v3.hymeko`, HELD_DWELL 6 | canonical v3 |
| success certifier | center_tol 0.02, settle_vel 0.06, dwell_req 6; grading `CoinRL4Dof._strict≥6 ∧ touched` | deployed strict-K6 |
| seed banks | dev transport/braking/settling_dwell (`config_sha 3ec6dbeb`) | frozen |
| D pilot | `receding_horizon/feedback_pilot_h30_result.json` | read, not re-run |

**Planner scorer (the crux):** `_lexo` = strict → dwell → −min_dtz → −min_speed → −effort. It has **no contact term and
no exit term.** This is the mechanical reason for the verdict below.

## 3 — Gold-baseline comparison (one harness, same 11 metrics, same strict-K6 grading)

Controllers A and D start from neutral (full episode); B and C from the reconstructed dev handoff (late phase). B vs C
is the apples-to-apples pair.

| metric | A (E-approach) | B (pi_0) | C (H=30 planner) | D (composed, neutral) |
|---|---|---|---|---|
| start | neutral | dev-handoff | dev-handoff | neutral |
| first contact | 9/9 | (≈handoff) | (≈handoff) | 9/9 (first_contact) |
| bilateral grasp | 5/9 | — | — | — |
| required contact retention | — | **0.474** | **0.176** | — |
| strict K=6 success | — | **0.194** | **0.645** | 7/9 delivered |
| max dwell | — | 1.48 | 3.87 | — |
| target entry | — | 0.516 | 0.710 | — |
| entry velocity (coin) | — | **0.049** | **1.073** | — |
| braking | — | 0.019 | 1.247 | — |
| first contact-loss step | — | 6.97 | 3.16 | — |
| target exit | — | **0.032** | **0.161** | 1/9 |
| total return | — | −25.0 | −22.4 | — |

D (the composed pipeline from neutral) delivers 7/9 with `contact_loss_after_acq` 5/9 (mostly legal push-and-coast after
placement) and only 1/9 target exit — a good headline. The qualification failure is specific to the late-phase
dev-handoff behavior, which the neutral headline does not expose.

## 4 — LOAD-BEARING: full-horizon H=30 teacher qualification (step 5, phase-conditional contact per step 6)

For every one of 31 disjoint dev late-start states, B (pi_0 continuation) and C (H=30 planner's **complete** replanned
continuation, M=1 feedback, full 30-step lookahead every step) were compared. Contact is required only **until stable
target entry** (`dtz≤0.02 ∧ settled`); loss after stable placement is legal.

| clause | measured (aggregate over 31) | pass? |
|---|---|---|
| (1) does not lose required contact | C req-contact 0.176 vs pi_0 0.474 (Δ **−0.30**); **4 new** required-contact losses | **FAIL** |
| (2) does not materially increase exit | C exit 0.161 vs pi_0 0.032 (Δ **+0.13**) | **FAIL** |
| (3) improves ≥1 of transport/brake/dwell/strict/return | strict 0.645 vs 0.194; dwell Δ **+2.39**; return Δ **+2.62** | **PASS** |

**Verdict: `H30_TEACHER_UNQUALIFIED: LOSES_REQUIRED_CONTACT, INCREASES_TARGET_EXIT`.**

### Mechanism (measured, not inferred)
The planner is a *strong* improver on the graded objective — strict success 3.3× pi_0, +2.4 dwell — **because** it drives
the coin into the zone hard: entry velocity 1.07 vs pi_0's 0.049 (22×), braking 1.25 vs 0.019, first contact-loss at
step 3.2 vs 6.97. With no contact or exit term in `_lexo`, the lexicographic optimum sacrifices required contact and
overshoots (higher exit) to buy strict/dwell. Per-state: C reaches strict in **20/31** states, but **14 of those 20**
sacrifice required contact.

### This closes last turn's open hypothesis
The chunk warm-start's contact collapse (0.19 vs pi_0 0.60) and `CHUNK_SUPERVISED_M1_FEEDBACK_NO_GAIN` are now explained:
the clone was **not** broken and was **not** mixed-teacher-averaging (both measured) — it faithfully reproduced an
**unqualified teacher** that itself does not preserve required contact. A student cannot be better-behaved than its
teacher. This is why step 7 forbids training a student from an unqualified teacher.

## 5 — Decision and the single next command

The teacher fails the full-rollout qualification ⇒ per step 7, **do not train a student.** Repair the planner objective
first with a lexicographic/constrained score:

```
1. preserve required (pre-stable-entry) robot-attributed contact
2. prevent target exit after entry
3. make target progress (min_dtz)
4. brake (settle speed)
5. settle and satisfy strict K=6
```

i.e. lift contact and exit ABOVE strict/dwell in `_lexo` (currently strict is the top key, contact/exit are absent), then
re-run this exact qualification harness until the planner qualifies. Only a qualified teacher advances to
`PHASE_CONDITIONED_FIRST_ACTION_DAGGER_V1` (step 8).

**SINGLE NEXT COMMAND:** implement `REPAIR_H30_PLANNER_OBJECTIVE_V1` — a constrained lexicographic scorer
(contact ≻ exit ≻ progress ≻ brake ≻ strict/K6) in the receding-horizon planner, then re-evaluate the planner itself
against this harness (`qualify_teacher`) until it returns `H30_TEACHER_QUALIFIED`. No student, no TD3/SAC, no chunk, no
final-test seeds, no task/success change.

## Claims / non-claims
**Claims:** (1) Identities verified (pi_0 `1902454c`, E `7dbbf1a7`, canonical H=30 config, deployed strict-K6 certifier).
(2) In one harness on 31 disjoint dev states the H=30 teacher improves the graded objective (strict 0.65 vs 0.19) but
violates required-contact preservation (0.18 vs 0.47, 4 new losses) and target-exit (0.16 vs 0.03) — UNQUALIFIED. (3)
The mechanism is the contact/exit-free lexicographic scorer driving hard, fast entry. (4) This explains the prior chunk
contact collapse as faithful cloning of an unqualified teacher.
**Non-claims:** NOT a single-state verdict (full 31-state aggregate, deterministic). NOT a claim that the *task* is
unsolvable — the repair (step 7) is the indicated path. NO training run occurred; D read from an existing artifact per
the cached-facts rule. No SAC/TD3/chunk/neutral-reset/final-test.

## Files
- impl: `hymeko_rl/coin_delivery/coin_baseline_reconstruction.py` (harness: RolloutTrace 11 metrics + phase-conditional
  contact, controller policies, `qualify_teacher`), `hymeko_rl/tests/test_coin_baseline_reconstruction.py` (7 tests).
- entry: `experiments/…/rl_entry/coin_baseline_reconstruction.py` (freeze + verify + A/B/C + read D + qualify),
  `experiments/…/rl_entry/plot_baseline_reconstruction.py`.
- results: `experiments/…/rl_entry/baseline_reconstruction_v1.json`, `…/baseline_reconstruction.svg`, this report.
- read (not re-run): `receding_horizon/feedback_pilot_h30_result.json` (D, SHA `1115ade3…`).

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; E_valselect `7dbbf1a7`; H=30 pop40/iters6/elite8;
reward v3 `galambos_task_deliver_v3.hymeko`; Mac (Apple Silicon), torch 2.12.0, mujoco 3.10.0. Deterministic (CEM fixed
seeds, pi_0 deterministic); B/C over 31 disjoint dev states, 81 s wall on 8 workers. No CORE.YAML items touched.
