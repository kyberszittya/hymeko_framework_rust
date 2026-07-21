---
title: Coin HyMeKo runtime recovery — reward now load-bearing; blocked at reward-eval alignment
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
verdict: COIN_HYMEKO_RECOVERY_BLOCKED
gates: HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS (reward now passes) · COIN_REWARD_EVALUATION_ALIGNMENT_PASS (FAIL)
---

# COIN_HYMEKO_RECOVERY_BLOCKED

The bundle-audit reward divergence is **repaired and proven load-bearing**, but the **reward↔evaluation alignment
gate fails** on a genuine three-way success-semantics contradiction in the spec bundle. Per §10.7/§10.8 training
stays blocked; no critic calibration, smoke, or campaign was launched. Reward-only repair is confirmed insufficient
(as the bundle audit predicted). This is a precise specification contradiction, not a training-variance finding.

## What was done (§10.1-10.3, §10.7)

- **§10.1 preserve** — branch `recovery/coin-hymeko-bundle-and-results` (from `acec5ad`), artifact root
  `experiments/2026_07_23_coin_hymeko_recovery/`, 19 rescue refs (`rescue/<sha>`), bundle hashes
  (combined `534e0c27042de8c5`). No historical commit amended.
- **§10.2 minimal repair (commit `3054572`)** — `DeliveryRLConfig.reward_source ∈ {hymeko_spec (default,
  authoritative) | python_delivery (explicit legacy ablation)}`; no implicit fallback (validated). `hymeko_spec`
  scores `inner.reward_spec.evaluate(inner, dtz, ctrl)` in `CoinDeliveryTrainEnv._transition`, mirroring
  `planar_grasp_env.py:837`. The old `delivery_reward` is retained only as the named legacy mode.
- **§10.3 sentinel — PASS** — v2b `zoneprog 10→200` now moves `rl_reward` (−1.51 → −0.91); `delivery_reward` legacy
  still available (−1.04). All **20 gate tests pass** under the new default. The `.hymeko` reward is now load-bearing.
- **§10.7 reward-alignment — FAIL** (the blocker below).

## The blocker (§10.11)

- **Failing gate:** `COIN_REWARD_EVALUATION_ALIGNMENT_PASS`.
- **Expected:** `strict_delivery > grasp_and_stall > zero_action` (full-horizon v2b return).
- **Actual:** `grasp_and_stall (−49.2) > strict_delivery (−57.4) > zero_action (−84.4) > oscillation (−94.3) > away (−145.7)`.
- **File / spec / line:**
  - `hymeko_rl/train/coin_full_action.py` — `FullActionDeliveryEnv.step` removes the center-terminal (`terminated =
    safety` only) so the 6-step strict dwell can accumulate.
  - `hymeko_rl/env/reward.py` — `_term_grasp_approach` / `_term_both_approach` are DENSE per-step penalties.
  - `data/robotics/galambos_task_deliver_v2b.hymeko` — `approach 4.0`, `bothapproach 4.0` (dense), `terminalgraded 30`.
  - `data/robotics/galambos_env.hymeko` — `success steps 5.0`.
  - `hymeko_rl/coin_delivery/delivery_certificate.py` — strict dwell = 6 steps.
- **Mechanism (per-term trace, seed 1011, reached_center=True):** `terminal_deliver_graded` DOES fire (+30.0), but
  `grasp_approach (−42.9) + both_approach (−45.0)` accumulate over the full 160-step horizon and swamp it (net
  −57.4). v2b's dense approach penalties are calibrated for a **terminate-at-success** episode (PlanarGraspEnv ends
  at the 5-step held-in-zone success). The full-action env removes that terminal to measure a 6-step strict dwell, so
  the penalties over-accumulate and grasp-and-stall (which sits still, accruing less approach penalty and never
  paying the transport cost) out-returns delivery.
- **Root contradiction:** THREE incompatible "success" definitions in the same runtime —
  1. v2b `terminal_deliver_graded` fires on **5-step held-in-zone** (`galambos_env` `success steps 5.0`);
  2. the residual/original `CoinDeliveryTrainEnv` terminates on **center-reach (1 step)** — *before* the 5-step dwell,
     so v2b's terminal bonus would never fire there either;
  3. the strict certificate requires a **6-step held dwell**.
- **Why more compute cannot help:** this is a reward-spec / episode-semantics contract mismatch. More seeds or steps
  optimize a misaligned objective; the optimum of the current bundle is grasp-and-stall, not delivery.
- **Reproduction:** `experiments/2026_07_23_coin_hymeko_recovery/logs/reward_alignment_v2b.json` (returns + per-term
  breakdown); rebuild via `make_full_action_env(fingertip_geometry="POINT", horizon=160)` (inherits `hymeko_spec`).

## Minimal next repair (proposed, NOT implemented — needs approval; it changes task semantics)

Reconcile the three success definitions into **one dwell contract** (the user's §7 architecture: "one strict
certificate contract → one matched evaluator"):

1. Pick a single held-dwell `K` (align `galambos_env.success steps`, the env terminal, and the strict certificate to
   the same `K`).
2. Terminate the RL episode on the **held-dwell completion** (not center-reach), matching v2b's
   `terminal_deliver_graded` firing condition — this bounds the dense approach penalty to the pre-delivery segment,
   restoring `delivery > stall`.
3. Re-run the §10.7 alignment gate; only if it passes, proceed to §10.4-10.10.

This is a **task-semantics change** (episode termination + dwell count), so per §10.7 ("do not silently modify task
semantics") it is reported, not applied, pending confirmation.

## Status of the §10 sequence

| step | status |
|---|---|
| 10.1 preserve | DONE |
| 10.2 reward repair | DONE (`3054572`) — reward load-bearing |
| 10.3 sentinel + bundle gate | reward sentinel PASS; full bundle gate not re-asserted (scene/robot still Python — DUPLICATE_EQUIVALENT / documentation-only per audit) |
| 10.7 reward-eval alignment | **FAIL → BLOCKER** |
| 10.4 historical reproduction | NOT STARTED (gated on 10.3/10.7) |
| 10.5 corrected bridge | NOT STARTED |
| 10.6 full-action BC | NOT STARTED |
| 10.8 critic calibration | NOT STARTED (blocked) |
| 10.9 SAC/TD3 smokes | NOT STARTED (blocked) |
| 10.10 campaign | NOT STARTED (blocked) |

## Preserved / unchanged

All historical results and quarantined commits are untouched (rescue refs created). The invalidated full-action RL
campaign remains `UNVERIFIED_FULL_ACTION_RL_RESULT_DUE_TO_REWARD_IDENTITY_MISMATCH`. The reward repair is on the
recovery branch only; `bde81ba` and the full-action branch are not amended.
