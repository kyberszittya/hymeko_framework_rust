---
title: Chunk M=1 execution-horizon diagnostic + mixed-teacher-averaging audit
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: M1_FEEDBACK_NO_GAIN + MIXED_TEACHER_AVERAGING_NOT_CONFIRMED
tags: [coin, action-chunk, receding-horizon, execution-horizon, mixed-teacher, audit, no-td3]
---

# CHUNK_SUPERVISED_M1_FEEDBACK_V1 — M=1 gives no gain; averaging NOT the cause; mixture NOT built

Two bounded, no-TD3 diagnostics on the **same** reproduced V2 supervised chunk actor (deterministic, seed 0; dataset 533
= 153 planner / 380 pi_0-fallback). Nothing was retrained to accommodate M=1 (item 7 — M is execution-only). No SAC, no
neutral-reset, no final-test seeds. CHUNK_TD3_V1 is **not** started.

## 1 — Execution horizon M=2 → 1 (only variable changed)

Same K=8 chunk, same actor, same disjoint dev states, same metrics; only the executed prefix changed. Receding-horizon
rollout on 31 disjoint dev states:

| metric | chunk **M=1** | chunk **M=2** | frozen pi_0 |
|---|---|---|---|
| contact retention | **0.105** | 0.188 | **0.596** |
| target exit (lower better) | 0.288 | 0.288 | 0.030 |
| max dwell | 0.55 | 0.84 | 1.48 |
| strict K6 | 0.065 | 0.129 | 0.194 |
| transport progress | −0.057 | −0.050 | — |

**Verdict: `M1_FEEDBACK_NO_GAIN`.** M=1 is not merely "no better" — it is **worse than M=2** on contact (0.105 vs
0.188), dwell, and strict. This **refutes** the V2 report's inferred limiter ("open-loop-M execution compounding imperfect
predictions"): if committing to 2 uncorrected steps were the problem, per-step replanning (M=1) would help. It hurts.
Replanning every step means the chunk's **first** action is executed at every step; if that first action is even slightly
off on the fragile bilateral-contact manifold, M=1 maximizes exposure to it while M=2 dilutes it with action[1]. So the
limiter lives in the **predicted first action**, not the horizon — exactly what item 8 tests.

## 2 — Mixed-teacher-averaging audit (item 8)

Two teachers label each state: exact **pi_0 fallback** vs **planner improvement**. Hypothesis: a regressor over two
dissimilar teachers that disagree in nearby states learns their **average** first action, worse than either. Measured on
the 164 base teacher-annotated states (both teacher first-actions recorded per state; pure/deterministic, no env):

| item-8 measurement | value | reads as |
|---|---|---|
| teacher first-action distance ‖pi0−planner‖ (median / p90) | 3.22 / 7.00 | teachers far apart (±4 scale) — **condition present** |
| local teacher-mode disagreement (kNN, k=8) | 0.387 | nearby states mix modes — **condition present** |
| conditional variance of first-action label in kNN | 0.681 | moderate local label noise — **condition present** |
| error vs mode-mixing correlation | +0.255 | error rises with mixing — corroborating |
| error by admissibility stratum (both / xor / neither) | 0.235 / 0.174 / 0.157 | error on improving-interior, **not** at the boundary |
| **learned lies BETWEEN the two teachers** | **0.055** | symptom **ABSENT** |
| **planner-state segment position** (1.0 = on planner target) | **0.947** | actor reaches the improving teacher — symptom **ABSENT** |
| planner-state fraction pulled toward pi_0 | 0.039 | almost never — symptom **ABSENT** |

**Verdict: `MIXED_TEACHER_AVERAGING_NOT_CONFIRMED`.** The *conditions* for averaging exist (far-apart teachers, mixed
neighborhoods, local label variance), but the *symptom* does not: the actor lies between the two teachers only 5.5% of the
time and sits at segment position 0.95 on planner-labeled states — it is a **faithful bimodal reproduction** of whichever
teacher is the label, not an average. Regression capacity was sufficient to fit both modes.

Figure: `experiments/2026_07_22_coin_v3_learning/rl_entry/chunk_m1_audit.svg` (contact bars + conditions-present/symptom-absent).

## 3 — Decision: EXACT_FALLBACK_CHUNK_MIXTURE_V1 NOT built

Item 9 is explicitly gated: *"If mixed-teacher averaging is confirmed, implement …"*. It is **not** confirmed, so the
mixture (frozen exact-pi_0 branch + planner-chunk actor + learned confidence head) is **not indicated** and was not built.
Building it would target a mechanism the discriminating audit shows is absent. Item 14's gate (M=1 clears **or** mixture
clears) is unmet → **CHUNK_TD3_V1 does not begin.**

## Mechanism — measured vs inferred vs hypothesis

- **Measured:** M=1 worse than M=2 (contact 0.105 < 0.188); the actor reaches its planner labels (segment 0.95) and does
  not average teachers (between 5.5%); first-action error 0.192/component concentrates on improving-interior states and
  correlates (+0.255) with local mode-mixing.
- **Inferred:** the contact collapse is **not** a first-action labeling/averaging artifact. The actor is faithful to its
  labels; the labels reproduce whichever teacher was chosen.
- **Still hypothesis (open, not decided here):** the collapse most plausibly lives in **the planner teacher itself** — its
  chunks were certified "safe-admissible" only over the executed M=2 prefix + a 15-step continuation window
  (`CONT_WINDOW`), not the full 60-step receding-horizon rollout. An improving chunk that preserves contact for 17 steps
  may trade it away later, and a faithful clone inherits that. The discriminating test (next, needs go): measure the
  **planner teacher's own** contact retention over the full 60-step horizon on dev. If the teacher also collapses contact,
  no supervised clone of this teacher can preserve it — the problem is the improvement teacher, not the learner.

## Claims / non-claims

**Claims:** (1) M=1 per-step replanning gives no gain and is worse than M=2 (contact 0.105 vs 0.188 vs pi_0 0.596) —
horizon is not the limiter. (2) The reproduced V2 actor does **not** average its two teachers (between-teachers 5.5%,
planner segment 0.95); mixed-teacher averaging is not confirmed. (3) Therefore the exact-fallback mixture is not indicated
and was not built; CHUNK_TD3_V1 not started.
**Non-claims:** NOT a TD3/SAC result. NOT that action-chunk control is dead. NOT that the planner teacher is proven the
culprit — that is the next discriminating test, not a verdict. Single deterministic reproduction (structural metrics, not
a stochastic RL ranking); the acceptance gates are unambiguous (contact −0.49; between 0.055).

## Files

- impl: `hymeko_rl/coin_delivery/coin_mixed_teacher_audit.py` (new), `hymeko_rl/tests/test_coin_mixed_teacher_audit.py`
  (new, 5 tests); `coin_feedback_chunk_v2.py` (+`record_teachers` flag, `build_teacher_annotated_dataset`,
  `reproduce_v2_actor` — canonical actor reproduction shared by both entries); `coin_chunk_td3.py`
  (`eval_receding_horizon`/`_rollout_rh` gained `m=` param).
- entries: `…/rl_entry/coin_chunk_m1_diagnostic.py`, `…/rl_entry/coin_mixed_teacher_audit.py`,
  `…/rl_entry/plot_chunk_m1_audit.py`.
- results: `…/rl_entry/chunk_m1_diagnostic.json`, `…/rl_entry/mixed_teacher_audit.json`, `…/rl_entry/chunk_m1_audit.svg`,
  this report.
- upstream: V2 warm-start report `2026-07-23-feedback-chunk-warmstart-v2.md`.

## Tests

`pytest -p no:randomly` (not slow): test_coin_chunk_td3 (6) + test_coin_feedback_chunk_v2 (2) + test_coin_mixed_teacher_audit
(5) = **13 passed**. Audit unit tests construct both regimes (averaging actor → CONFIRMED; on-target actor →
NOT_CONFIRMED) and enforce preconditions. `ruff --select F,E9` clean on all touched files.

## Provenance

Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 sha `1902454c`; Mac (Apple Silicon), torch 2.12.0, mujoco 3.10.0.
Deterministic (CEM fixed seeds; supervised seed 0) — the reproduced actor gives identical M=1 numbers on re-run
(0.1045 / 0.1882 / 0.5962). No §6.5 anti-patterns introduced (record_teachers is an additive flag, not a new code path;
reproduce_v2_actor removes the m1/audit reproduction duplication). No planner during learned-policy evaluation.
No CORE.YAML items touched.
