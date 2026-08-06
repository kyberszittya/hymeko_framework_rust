---
title: REPAIR_H30_PLANNER_OBJECTIVE_V1 — feasibility-gated planner still unqualified (floor-robust)
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: H30_TEACHER_UNQUALIFIED — OBJECTIVE_REPAIR_INSUFFICIENT (contact/delivery in tension under CEM replanning)
tags: [coin, planner-repair, feasibility-constraint, lexicographic, teacher-qualification, no-training]
---

# REPAIR_H30_PLANNER_OBJECTIVE_V1 — objective repair does not qualify the teacher (floor-robust)

No training. The H=30 planner was repaired exactly as specified: **phase-conditional feasibility constraints gating a
lexicographic task objective** (strict-K6 kept high in the tuple, never last). Re-run through the identical 31-state
qualification harness at two contact floors, it **still fails** the contact and exit clauses. The failure is
**floor-robust** — lowering the floor makes contact *worse*, not better — so it is not a threshold artifact. No
student/TD3/SAC/chunk; no final-test seeds.

## What was built (contract steps 1–7)
`hymeko_rl/coin_delivery/coin_planner_repair.py` (the proven `coin_v3_receding_horizon` scorer is left byte-untouched;
this is an additive strategy reusing its CEM primitives):

- **Two phase-conditional feasibility constraints** (never task terms): `premature_required_contact_loss` (pre-boundary
  contact retention below a floor — a *material* abandonment before the frozen completion boundary; release AFTER the
  boundary is legal) and `illegal_target_exit` (entered then exited **before strict-K6**). Both boundary definitions
  measured: **A** = first stable target entry, **B** = strict-K6 certification. Separately tracked: first entry, first
  stable entry, exit after entry, exit after stable entry, exit before K6 (step 2).
- **Candidate admissibility** (step 3): infeasible if either violation; a feasible candidate is never beaten by an
  infeasible one; if all infeasible, the **least-violating** is chosen and `ALL_CANDIDATES_INFEASIBLE` flagged.
- **Lexicographic task objective among feasible** (step 4): `(feasible, any_strict, max_dwell, −min_dtz,
  −excess_entry_speed, −effort)` — strict-K6 stays high, so the repair cannot yield a safe-but-non-delivering planner.
- Raw contact duration after placement never rewarded (step 5; in-plan rollout stops at K6). **Kept unchanged** (step 6):
  horizon 30 / pop 40 / iters 6 / elite 8, CEM seeds, dev handoff bank, reward, certifier, task, routing.
- **7 deterministic ordering tests** (step 7) all pass: early-contact-break loses to safe-strict; fast-enter-exit loses
  to stable-entry; release-after-K6 not penalised; all-infeasible ⇒ least-violating + flag; strict outranks progress.

## Result — floor-robust UNQUALIFIED (contract steps 8–9)

Identical 31-state harness, frozen boundary **A (stable_entry)** — chosen because exit<K6 was identical under A and B
(Δ 0.0), so the minimal physical requirement (release after settling is legal) is the correct one; B only tightens.

| aggregate (vs pi_0) | floor 0.75 | floor 0.50 | pi_0 | clause |
|---|---|---|---|---|
| required-contact retention | 0.198 | **0.132** | 0.474 | **FAIL** (Δ −0.28 / −0.34; 4 new losses) |
| target exit before K6 | 0.129 | 0.129 | 0.032 | **FAIL** (Δ +0.10, floor-invariant) |
| strict K6 | 0.516 | 0.355 | 0.194 | advantage PASS |
| max dwell Δ | +1.61 | +0.65 | — | — |
| ALL_CANDIDATES_INFEASIBLE freq | 0.471 | 0.429 | — | — |
| first-action stability (cosine) | 0.28 / −0.16 / 0.15 | 0.15 / −0.07 / −0.10 | — | unstable |

**Verdict: `H30_TEACHER_UNQUALIFIED: REPAIRED_LOSES_REQUIRED_CONTACT, REPAIRED_INCREASES_EXIT_BEFORE_K6`** — named
mechanism **OBJECTIVE_REPAIR_INSUFFICIENT (floor-robust)**.

## Mechanism (measured, with the discriminating test)
1. **Floor-robust, ruling out a miscalibrated constraint.** The 0.75 floor sits above pi_0's own 0.474 average, so a
   natural suspicion is "too strict." The discriminating test (floor 0.50, pi_0-calibrated) **refutes** it: contact got
   *worse* (0.198 → 0.132) and delivery worse (0.516 → 0.355). A laxer floor admits contact-breaking candidates, so the
   task objective re-selects the delivering-but-low-contact plan — the constraint bites less, not more. This is the
   opposite of a threshold artifact.
2. **Contact and delivery are in tension** (both floors): delivering states hold *less* contact than non-delivering ones
   (0.170 vs 0.229 @0.75; 0.088 vs 0.156 @0.50). The single best-contact state (0.98) does **not** deliver.
3. **Lookahead feasibility ≠ executed contact.** The constraint gates the 30-step *simulated* candidate, but only the
   **first** action is executed then the planner replans. Even the 11/31 states with feasible candidates available
   (infeasibility < 0.2) had executed contact only 0.309. Re-ranking the lookahead cannot govern the first-action-only
   executed path — which is also **unstable** (first-action cosine ≈ 0.15, sometimes negative).

## Interpretation — measured vs inferred vs open
- **Measured:** objective repair (feasibility-gated candidate re-ranking over the same CEM candidate distribution) does
  not qualify the H=30 teacher, at two contact floors; contact 0.13–0.20 ≪ pi_0 0.47; delivering states hold less
  contact; ~43–47% of replans have no feasible candidate; the repaired first action is unstable.
- **Inferred:** within the CEM-over-4-DoF-arm-actions replanned every step, delivery progress and required-contact
  preservation are antagonistic, and *ranking* candidates cannot separate them because feasible-and-delivering plans are
  scarce and lookahead-feasibility does not constrain the executed first action.
- **Open (NOT a closed verdict):** this does not prove contact-preserving delivery is impossible. pi_0 preserves more
  contact (0.47) but under-delivers (0.19); a controller class that constrains the **executed action's immediate
  contact effect** (not just 30-step lookahead feasibility), or that is not CEM-replanning at all, is untested. The
  tension may also be the arc's kinematic contact-mechanics wall. This is a direction, not a conclusion.

## Decision
The objective repair does not qualify the teacher and the evidence is floor-robust, so **continuing to tune the scorer /
floor is a dead lever** — I stopped rather than chase it (§ operating principles). Per the campaign, a student is still
**not** trained (no qualified teacher). No TD3/SAC/chunk/final-test. Awaiting direction on the next lever (candidate-space
or executed-action-constraint change), which is a new structural decision, not a re-tune.

## Claims / non-claims
**Claims:** (1) The repaired planner implements phase-conditional feasibility constraints + a strict-high lexicographic
objective, verified by 7 ordering tests. (2) It is UNQUALIFIED on contact and exit-before-K6 at floors 0.75 and 0.50 —
floor-robust. (3) The mechanism is a genuine contact/delivery tension plus a lookahead-vs-executed gap and first-action
instability, not a miscalibrated threshold (discriminating floor test on record).
**Non-claims:** NOT a first-pass result (two full 31-state runs, deterministic, plus a floor discriminating test). NOT a
claim that contact-preserving delivery is impossible — untested controller classes remain. No training occurred.

## Files
- impl: `hymeko_rl/coin_delivery/coin_planner_repair.py` (FeasibilityConfig, classify_feasibility, score_candidate,
  repaired_key, select_candidate, plan_first_action_repaired, RepairedPlannerPolicy, repaired_first_action_stability,
  final_qualification), `hymeko_rl/tests/test_coin_planner_repair.py` (7 tests); `coin_baseline_reconstruction.py`
  (+additive `exit_before_k6`/`k6_step` in RolloutTrace).
- entry/plot: `experiments/…/rl_entry/coin_planner_repair_qualify.py`, `…/plot_planner_repair.py`.
- results: `…/planner_repair_qualify_v1.json` (floor 0.75, canonical), `…_floor075.json`, `…_floor05.json`,
  `…/planner_repair.svg`, this report.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; canonical H=30 pop40/iters6/elite8; reward v3;
strict-K6 certifier; frozen 31-state dev bank (`config_sha 3ec6dbeb`). Deterministic (CEM fixed seeds, pi_0
deterministic). Two full 31-state runs (~190 s each, 8 workers) + floor discriminating test. No CORE.YAML items touched;
proven `coin_v3_receding_horizon` scorer byte-untouched.
