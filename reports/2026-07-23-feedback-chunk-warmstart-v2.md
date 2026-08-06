---
title: Feedback chunk warm-start V2
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: FEEDBACK_CHUNK_WARMSTART_V2_STILL_UNDERPERFORMS
tags: [coin, action-chunk, receding-horizon, feedback-planner, dagger, warm-start, no-td3]
---

# FEEDBACK_CHUNK_WARMSTART_V2_STILL_UNDERPERFORMS — CHUNK_TD3_V1 not started

Dense feedback-planner warm-start (V2) — no TD3. The V1 warm-start (59 isolated CEM/pi_0 chunks) is replaced by DENSE
sampling of the proven H-receding-horizon **feedback** planner at every replanning state, with executed-prefix
admissibility, pi_0 fallback, prefix-weighted regression, and supervised DAgger. It **improves** on V1 but still does
not clear the §10 acceptance gate, so per §11 CHUNK_TD3_V1 does not begin.

## What V2 built (verified)
- **Dense feedback teacher** (`feedback_chunk_rollout`): at every replanning state of a receding-horizon feedback
  rollout, record `(state_t, planner K=8 chunk, executed M=2 prefix, state_{t+M}, phase/contact/event, outcomes)` —
  **one example per replan**, not one per trajectory. Planner is WARM_START_ONLY (no planner at eval, asserted).
- **Executed-prefix admissibility (§4):** a planner chunk is labeled `planner` only if its **M=2 prefix** preserves
  robot contact vs pi_0 and doesn't induce target exit **and** the planner continuation improves ≥1 of
  {progress, braking, dwell, strict, entry} without degrading contact/exit; else the **pi_0 chunk is the safe fallback**.
  Full-K contact reported but not required (unused suffix discarded).
- **Dataset:** base 164 (51 planner-improving / 113 pi_0-fallback; phase transport 134 / braking 20 / settling 10);
  + 2 DAgger rounds → **533 examples** (153 planner / 380 pi_0-fallback), sha `d2d4bc2b`. Trajectory-level disjoint
  train/dev.
- **Prefix-weighted regression** (frozen weights [1,1,0.5,0.5,0.25,0.25,0.1,0.1]) + **supervised DAgger** (roll learned
  actor, query planner offline, add safe label, retrain). Tested.

## Result — improved over V1, still fails the gate
Receding-horizon eval (V2 chunk vs frozen pi_0) on 31 disjoint dev states:

| metric | V1 chunk | V2 chunk | pi_0 | V2 Δ vs pi_0 |
|---|---|---|---|---|
| contact retention | 0.10 | **0.19** | 0.60 | **−0.408** |
| max dwell | 0.97 | 0.84 | 1.48 | −0.65 |
| strict K6 | 0.13 | 0.13 | 0.19 | −0.065 |
| target exit | 0.23 | 0.29 | 0.03 | +0.258 |

V2 lifts contact retention 0.10 → 0.19 (Δcontact −0.496 → −0.408) — the dense/safe-fallback warm-start helps — but it
still **materially underperforms pi_0** on contact/dwell/exit. **Acceptance gate: FAIL** (contact −0.408 ≪ −0.05).

## Mechanism (measured vs inferred)
- **Measured:** the supervised chunk MSE **rose** after DAgger diversification (base first-action MSE 0.009 → final
  0.032; per-index 0.032→0.050) — the dense feedback distribution is harder to fit, so the predicted M=2 actions are
  ~0.18/component off.
- **Inferred:** the failure is **open-loop-M execution**, not the warm-start data. Even with 71% safe pi_0-fallback
  labels, a chunk actor whose predictions are imperfect (and OOD on dev), executed **2 steps open-loop before
  replanning**, breaks the fragile bilateral contact that pi_0's **per-step** feedback holds. The chunk lookahead is
  fine; committing to 2 uncorrected steps is the problem.

## Decision
§10 gate fails ⇒ per §11, **CHUNK_TD3_V1 is not started** (would run from a contact-collapsing warm-start, which §12
forbids). V2 is a real improvement but not sufficient.

## Claims / non-claims
**Claims:** (1) Dense feedback sampling + executed-prefix admissibility + pi_0 fallback + prefix-weighted DAgger builds a
533-example, phase-balanced, safely-labeled warm-start (verified). (2) The V2 supervised chunk baseline improves contact
0.10→0.19 vs V1 but still underperforms pi_0 (contact −0.408) — gate fails. (3) The limiter is open-loop-M=2 execution
compounding imperfect predictions, not the warm-start density.
**Non-claims:** NOT that action-chunk control is dead — the diagnosis points at M, not the chunk. NOT a TD3 result. No
neutral-reset/final-test/SAC; no planner at eval.

## Next narrow experiment (needs your go)
The measured limiter is M, not the data. Set **M=1** (execute one action, replan every step — pure feedback; the K=8
chunk is retained only for lookahead/critic) and re-evaluate the supervised baseline: if contact recovers toward pi_0,
the chunk critic can still exploit the K-step lookahead in TD3 while execution stays per-step feedback. Alternatively add
a **contact-guarded execution** (abort the suffix and replan the instant contact is lost). Only once the supervised
chunk baseline clears the acceptance gate does CHUNK_TD3_V1 begin.

## Files
- impl: `hymeko_rl/coin_delivery/coin_feedback_chunk_v2.py`, `hymeko_rl/tests/test_coin_feedback_chunk_v2.py`,
  `experiments/…/coin_feedback_chunk_warmstart_v2.py`.
- results: `experiments/…/feedback_chunk_warmstart_v2.json`, this report.
- upstream: chunk contracts `e94828a`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Mac, torch 2.12.0, mujoco 3.10.0. Deterministic
(CEM fixed seeds; supervised seed 0); reproduced identically on re-run. No planner during evaluation.
