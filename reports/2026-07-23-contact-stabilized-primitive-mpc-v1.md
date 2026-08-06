---
title: CONTACT_STABILIZED_PRIMITIVE_MPC_V1 — 10-D primitive + frozen pi_0 + immediate contact guard
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: PRIMITIVE_MPC_UNQUALIFIED — LOSES_REQUIRED_CONTACT (closest of the arc; tension localised to braking)
tags: [coin, primitive-mpc, execution-guard, frozen-pi0, teacher-qualification, no-training]
---

# CONTACT_STABILIZED_PRIMITIVE_MPC_V1 — the contact guard works, but delivery still costs contact in the braking phase

No training. A new structural baseline: keep pi_0 immutable as the per-timestep feedback law, parameterise the
improvement as ONE 10-D push→brake→settle primitive θ, run it closed-loop with LIVE mode detection, and gate every
non-pi_0 action through an **immediate contact-preserving execution guard** (2-step effect vs a pi_0 reference,
line-searched toward pi_0). Optimise θ (not raw actions) by bounded CEM, re-planned every 4 low-level steps. This is the
"constrain the executed action's immediate contact effect" lever the raw/repaired H=30 planners lacked.

It is the **closest** controller of the whole arc — it clears 3 of 5 qualification clauses and is the only one that both
beats pi_0 on strict and stays near pi_0 on contact — but it still **materially loses required contact**, so it does not
qualify. No student/TD3/SAC/PPO; no final-test seeds.

## What was built (contract steps 1–8)
`hymeko_rl/coin_delivery/coin_primitive_mpc.py`: `Theta` (10-D: delta_push[4], delta_brake[4], brake_distance,
settle_speed; frozen bounds ±1.5 / [0.02,0.12]); `detect_mode` (PUSH/BRAKE/SETTLE from live state, never elapsed time);
`execution_guard` (pi_0 reference; largest α∈{1,.75,.5,.25,0} with `pi_0+α·offset` not losing contact pi_0 keeps nor
causing an illegal exit; else α=0 fallback; release after certified placement legal); `PrimitiveController` (the SAME
controller used in candidate simulation and real execution); `simulate_primitive` (full closed-loop θ scoring, feasibility
+ task fields); `plan_theta` (bounded CEM over 10-D, NOT raw actions); `PrimitiveMPCPolicy` (receding-horizon, replan θ
every 4 steps, stores θ never an action sequence). **11 deterministic contract tests pass** (θ 10-D; actions ∈[−4,4];
SETTLE ≡ pi_0; state-based transitions; guard α=0 ≡ pi_0; unsafe blended toward pi_0; release-after-placement not
penalised; same controller sim≡real; deterministic restoration; no open-loop sequence stored; strict outranks progress).
Kept unchanged (step 6): CEM pop40/iters6/elite8/horizon60, replan-4, guard-2, reward, certifier, task, routing.

## Result — one bounded 31-state development evaluation

| aggregate (vs pi_0) | pi_0 | primitive MPC | clause |
|---|---|---|---|
| required-contact retention | 0.474 | **0.363** (Δ −0.112, 3 new losses) | **FAIL** |
| exit before K6 | 0.032 | 0.097 (Δ +0.065) | **FAIL** |
| strict K6 | 0.194 | 0.258 (Δ +0.065) | advantage PASS |
| max dwell Δ | — | +0.23 | — |
| guard fallback rate | — | 0.140 | not-fallback-dependent PASS |
| first-primitive θ std / range | — | 0.162 | cem-stable PASS |

**Verdict: `PRIMITIVE_MPC_UNQUALIFIED: PRIMITIVE_MPC_LOSES_REQUIRED_CONTACT`** (the exit-before-K6 clause also fails,
marginally). Mode occupancy: BRAKE 0.75 / SETTLE 0.25 / PUSH 0.00 (the late handoff states never enter PUSH); guard
intervention 0.68.

### The tension is localised to the braking phase (per-family)
| family | n | pi_0 contact | primitive contact | pi_0 strict | primitive strict | guard_fb |
|---|---|---|---|---|---|---|
| transport | 15 | 0.612 | 0.549 | 0.00 | 0.00 | 0.03 |
| braking | 10 | 0.552 | **0.301** | 0.10 | **0.30** | 0.09 |
| settling_dwell | 6 | 0.00 | 0.00 | 0.83 | 0.83 | 0.50 |

- **transport (far):** the guard nearly matches pi_0's contact (0.55 vs 0.61) at 3% fallback — no delivery is available
  and none is forced. The guard preserves contact exactly as intended.
- **braking (near — the decisive regime):** the primitive **triples** delivery (0.30 vs 0.10) but **only** by trading
  contact (0.55 → 0.30). The whole aggregate contact deficit comes from here.
- **settling_dwell (at target):** identical to pi_0 (SETTLE ≡ pi_0) — a correct no-op.

## Mechanism (measured)
1. **The immediate guard genuinely preserves contact** where the raw/repaired H=30 planners could not: aggregate contact
   0.36 vs their 0.18, at only 14% fallback (so the gain is not pi_0-fallback in disguise — the `not_fallback_dependent`
   clause passes). On transport it is within 0.06 of pi_0.
2. **But delivery in the braking regime is coupled to contact loss.** To push the coin into the zone the primitive must
   emit an action that breaks contact within the 2-step guard window. The guard admits it because pi_0's *own* contact is
   already dropping there (0.55), so "not worse than pi_0" is satisfied — and the coin gets delivered (strict 0.10→0.30)
   at the cost of contact (0.55→0.30). The guard blocks contact loss pi_0 would have avoided; it cannot manufacture a
   contact-preserving delivery action that does not exist in the braking geometry.
3. **The CEM over 10-D θ is stable** (θ std/range 0.16) — the low dimensionality fixed the instability the 120-D raw
   planner showed (first-action cosine ≈ 0.15 there).

## Interpretation — measured vs inferred vs open
- **Measured:** the primitive+guard is the arc's only controller that improves strict over pi_0 (+0.065, dwell +0.23)
  while keeping contact near pi_0 (0.36 vs planners' 0.18); it clears advantage / not-fallback-dependent / cem-stable but
  fails contact (−0.11) and exit-before-K6 (+0.065). Pareto-superior to the planners, not dominant over pi_0.
- **Inferred:** the residual contact deficit is not a guard failure or a threshold artifact — it is physically localised
  to the **braking phase**, where delivering the coin requires a contact-breaking push and pi_0 itself is already losing
  contact, so the guard permits it.
- **Open (NOT closed):** the primitive delivered with contact ≥ 0.5 on 2/31 states — contact-preserving delivery is not
  impossible, only rare in the braking geometry. Untested levers that could close the −0.06 contact gap: a **longer /
  contact-weighted guard horizon** in the braking phase (2 steps may be too short to see a re-grasp), a **brake_distance /
  entry-speed bound** that enters the zone slower (gentler push → less contact break), or a braking-specific offset that
  decelerates *through* contact rather than releasing. These are primitive/guard refinements, not a new class.

## Decision
UNQUALIFIED ⇒ per the campaign, **no student is trained** (PRIMITIVE_POLICY_BC/DAgger/SAC/TD3 are gated on
PRIMITIVE_MPC_QUALIFIED, which is not reached). No TD3/SAC/PPO/final-test. The result is the closest yet and localises the
remaining wall to the braking phase — a concrete next lever, awaiting direction.

## Claims / non-claims
**Claims:** (1) A 10-D primitive + frozen pi_0 + immediate contact guard is implemented and verified by 11 contract tests.
(2) On 31 dev states it preserves contact far better than the H=30 planners (0.36 vs 0.18) at low fallback (0.14) and
shows a modest strict/dwell advantage over pi_0, but still loses required contact (0.36 vs 0.474) — UNQUALIFIED. (3) The
deficit is localised to the braking regime, where delivery is coupled to contact loss.
**Non-claims:** NOT a claim that contact-preserving delivery is impossible (2/31 states did it). NOT a first-pass single
measurement (full 31-state, deterministic CEM). No training occurred. B (raw H=30) is the cached reconstruction row, not
re-run.

## Files
- impl: `hymeko_rl/coin_delivery/coin_primitive_mpc.py`, `hymeko_rl/tests/test_coin_primitive_mpc.py` (11 tests).
- entry/plot: `experiments/…/rl_entry/coin_primitive_mpc_qualify.py` (slice+merge; serial per slice — the mp.Pool proved
  unstable for this heavy workload, killed by BLAS thread oversubscription), `…/plot_primitive_mpc.py`.
- results: `…/primitive_mpc_qualify_v1.json`, `…/primitive_mpc_slice_{0..3}.json`, `…/primitive_mpc.svg`, this report.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c` (immutable); reward v3; strict-K6 certifier; frozen
31-state dev bank (`config_sha 3ec6dbeb`). Deterministic (CEM fixed seeds, pi_0 deterministic). Run as 4 independent
single-process slices (~4–7 min/state under contention) + merge. No CORE.YAML items touched; proven `coin_v3_receding_horizon`
scorer untouched.
