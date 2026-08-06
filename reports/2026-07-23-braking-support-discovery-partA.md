---
title: CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 Part A — braking action-support discovery
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: BRAKING_SAFE_BENEFICIAL_SUPPORT_INSUFFICIENT — Part B not started
tags: [coin, braking, action-support, branch-search, gate, no-training]
---

# CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 — Part A: safe-beneficial braking support is real but too sparse

No training. Part A of the V2 follow-up asks a single gating question before building any state-feedback braking basis:
**at the braking states the frozen pi_0 visits, does a bounded offset to the exact pi_0 action exist that decelerates the
target-directed coin velocity while preserving required contact?** The frozen thresholds and contracts were committed
before measurement (`355437fc`). pi_0, reward, certifier, task, and the V1 primitive are all frozen.

## Protocol (frozen, committed before measuring)
- **Braking snapshot bank:** drove each of the 31 dev handoffs forward with frozen pi_0 (V1 mode detection), recording the
  ABSOLUTE pi_0-from-reset step of the first BRAKE state and every 5th thereafter → **272 braking states** (manifest sha
  `932ec4dd…`). The absolute step (not a MuJoCo snapshot) is stored so each state reconstructs EXACTLY via `replay_pi0` —
  a raw snapshot would not reproduce the `node_features` velocity buffer.
- **Branch search:** at each state, 32 deterministic bounded candidate offsets (‖Δ‖≤1.2) around pi_0; each candidate ran
  4 low-level steps (candidate at step 0, frozen pi_0 after) with exact state restoration vs the pi_0-only branch. No
  open-loop raw-action sequence.
- **SAFE_BENEFICIAL (frozen tolerances):** no new required-contact loss vs pi_0; no new exit-before-K6; reduces peak
  target-directed coin velocity by ≥ 0.02 m/s OR gains strict; final distance-to-zone within 0.005 m of pi_0.
- **Gate (preregistered):** FOUND only if ≥ 50 % of braking states have ≥1 safe-beneficial candidate AND ≥ 3 *failing*
  (contact-losing / exit) states have support.

## Result — support exists, spread across failure classes, but below the bar

| V1 outcome class | n braking states | with ≥1 safe-beneficial | mean safe/state |
|---|---|---|---|
| contact_losing | 234 | 46 (20%) | 0.51 |
| target_exit | 27 | 6 (22%) | 0.74 |
| delivered_contact_preserving | 11 | 4 (36%) | 0.73 |
| **overall** | **272** | **56 (21%)** | 0.54 |

- **21 % overall** with safe-beneficial support — below the preregistered 0.5 bar.
- Support is **not** isolated to the successful states: 52 failing states have support (≥3 requirement met) — but the
  aggregate fraction gates the verdict.
- Real deceleration where support exists: median 0.046, p90 0.10, max 0.21 m/s.

**Verdict: `BRAKING_SAFE_BENEFICIAL_SUPPORT_INSUFFICIENT`.** Per the protocol, **Part B is not started** and the primitive
is not tuned blindly.

## Mechanism (measured) — support lives only where the coin approaches fast
The discriminator between supported and unsupported braking states is **not distance** (both dtz median ≈ 0.09 m) but
**speed:** supported states have pi_0 radial (target-directed) coin velocity ≈ **0.096 m/s** vs **0.027 m/s** for
unsupported. Safe deceleration is only *available* when there is excessive target-directed velocity to remove; at the
majority of braking states the coin is already moving slowly, so no bounded offset both decelerates and stays safe — and
(from V1) no offset gains strict without breaking contact. The braking primitive's value is therefore confined to the
minority fast-approach states, which is why a state-feedback braking basis cannot remove the aggregate contact deficit.

## Interpretation — measured vs inferred vs open
- **Measured:** 21 % of 272 braking states admit a safe-beneficial bounded offset (decelerate ≥0.02 m/s, keep contact,
  keep progress); support concentrates at high-radial-velocity states (0.096 vs 0.027 m/s); best deceleration 0.21 m/s.
- **Inferred:** the V1 contact/delivery coupling has a velocity signature — only fast-approach braking states have room to
  decelerate safely; the rest are already slow, so the primitive has no safe lever there.
- **Open (not a verdict):** the 21 % is under THIS frozen bounded config (32 candidates, 4-step horizon, ‖Δ‖≤1.2, and a
  tight 0.005 m progress tolerance that penalises the small progress cost of decelerating). A denser candidate set, a
  longer branch horizon, or a looser progress tolerance would raise the number — but the mechanistic concentration at
  fast-approach states means it is unlikely to clear 0.5, and relaxing a frozen gate to pass is exactly the blind tuning
  the protocol forbids. If the objective is revised to *target the fast-approach subset only*, support there is real
  (best decel 0.21 m/s) — that would be a re-scoped campaign, a user decision.

## Decision
Gate fails ⇒ **Part B (structured feedback braking basis, BC/DAgger/SAC/TD3) does not run.** No student, no RL, no
final-test seeds. The result cleanly prevents fitting a braking basis that would only help ~1 in 5 braking states.

## Claims / non-claims
**Claims:** (1) A deterministic braking-state bank (272 states, exact `replay_pi0` reconstruction) + bounded branch-search
support test with frozen tolerances is built and committed pre-measurement. (2) Safe-beneficial braking support exists at
21 % of braking states, spread across failure classes, with real deceleration (≤0.21 m/s), but is below the preregistered
0.5 bar ⇒ INSUFFICIENT. (3) Support concentrates at high-radial-velocity (fast-approach) states.
**Non-claims:** NOT that braking support is zero (56 states have it). NOT a claim under a denser/looser config. No training
occurred; the V1 primitive and pi_0 are unchanged.

## Files
- impl: `hymeko_rl/coin_delivery/coin_braking_support.py`, `hymeko_rl/tests/test_coin_braking_support.py` (10 tests).
- entry/plot: `experiments/…/rl_entry/coin_braking_support_partA.py`, `…/plot_braking_support.py`.
- results: `…/braking_snapshot_manifest.json` (sha `932ec4dd…`), `…/braking_support_partA.json`, `…/braking_support.svg`,
  this report.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c` (immutable); frozen 31-state dev bank
(`config_sha 3ec6dbeb`); V1 primitive `77e45418`. Deterministic (fixed candidate seed, `replay_pi0` exact
reconstruction). Contracts + frozen thresholds committed `355437fc` before measuring. No CORE.YAML items touched.
