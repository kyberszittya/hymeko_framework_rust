---
title: Residual hold-horizon leverage sweep V1 (label-only)
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: RESIDUAL_SIGNAL_INCREASES_WITH_HOLD_HORIZON
tags: [coin, residual, temporal-leverage, counterfactual, preregistered, positive]
---

# RESIDUAL_SIGNAL_INCREASES_WITH_HOLD_HORIZON

**Label-only diagnostic — no critic trained, no actor updated, no SAC, gate/reward/residual-bound unchanged.**
Preregistered: config frozen at commit `8d85923` (state manifest sha `c9ea1bea`, candidate manifest sha `9fd0d093`)
**before** the run (commit C is the results). Determinism ×2: **True** (all 22 400 hold branches reproduced identically).

## Question
Does the local counterfactual return signal ΔG become measurable when the same bounded residual acts for K consecutive
gate-active steps before the frozen `pi_0` continuation? (The `RESIDUAL_CRITIC_ROUTE_BLOCKED` finding attributed the
critic failure to a one-step residual having near-zero leverage on the full-horizon return.)

## Design
40 captured gate-active states (10 / family: transport, entry, settling, contact_retention). 56 deterministic
candidates / state = {0, .01, .025, .05, .10, .25} × {signed actuator bases (8), frozen isotropic (3, seed 20260723)},
IDENTICAL across every K and paired by `state_group_id`. K ∈ {1, 2, 4, 8, 16}. Each state's complete controller state
(planar MuJoCo buffers + wrapper counters + a deepcopy of the StableEngagementGate FSM) is restored; the residual is
held for ≤K gate-active steps (gate-off ⇒ `pi_0` bit-identical), then frozen `pi_0` continues to term/trunc; canonical
γ=0.99 discounted return. Every branch run twice, required identical. Step-0 base uses the captured `g.base` (a restore
does not reproduce the `node_features` velocity buffer → reading `rl.obs()` at t0 is non-reproducible over the chaotic
contact horizon; the buffer is correct after the first `mj_step`).

## Result — leverage rises monotonically with K
Per-group median |ΔG|, paired bootstrap vs K=1 (4000 resamples by `state_group_id`):

| K | median leverage | Δ vs K=1 (mean) | 95% CI | gate (ci_low>0) |
|---|---|---|---|---|
| 1 | 0.700 | +0.000 | [0.00, 0.00] | — |
| 2 | 0.889 | **+1.903** | [0.174, 5.149] | ✅ |
| 4 | 0.891 | **+2.115** | [0.316, 5.462] | ✅ |
| 8 | 0.897 | **+2.657** | [0.486, 6.296] | ✅ |
| 16 | 0.941 | **+3.405** | [0.891, 7.215] | ✅ |

Every K≥2 clears the frozen leverage gate (paired ci95 excludes 0 above), monotonically. Median |ΔG| by family (K=1→16):
transport 0.45→0.75, **entry 0.21→1.10 (5×), settling 0.09→0.92 (10×)**, contact_retention 1.10→2.12. frac(|ΔG|≥5)
rises in every family (settling 0.10→0.31). `median_eff_clip_loss = 0.000` at all K (the ±0.25 residual on top of `pi_0`
never saturates the ±4 bound — authority fully expressed, not eaten by clipping). `best_action_stable = 1.000`.

## Mechanism (measured vs inferred)
- **Measured:** a residual held longer moves the canonical return more — the one-step near-zero signal that blocked the
  critic becomes resolvable by K≥2, growing to K=16. The gain is heterogeneous (median leverage rises only 0.70→0.94,
  but the mean paired diff is +1.9…+3.4 with CI up to 7.2 — a heavy tail: a subset of states gains large leverage).
- **The asymmetry (measured, load-bearing caveat):** the extra leverage is predominantly **destabilizing**. Harmful
  candidate fraction > beneficial at every K (K=16: 0.300 vs 0.222) and both grow with K; **strict-success gain is
  negative and worsens** (−0.027 → −0.072); P(contact break) ≈ 0.94→0.97 across all K. So the measurable K-hold signal
  is dominated by "which residuals wreck the delivery," not "which residuals improve it."
- **Inferred:** the `ROUTE_BLOCKED` blocker was indeed credit-assignment horizon — one-step leverage is below
  resolution; multi-step leverage is resolvable. But a K-hold advantage critic would mostly learn a **harm-rejection /
  safety** signal; genuine *improvement* candidates stay a minority.

## §13 gate — ADVANTAGE_COMPOSITE_CRITIC_V3 is UNLOCKED (not run here)
The signal increase is **preregistered and reproducible**, so per §13 an eligible K exists — the frozen leverage gate
passes at K∈{2,4,8,16}; K=16 has the strongest, tightest-above-zero CI (ci_low 0.891) with monotone support. Per §8/§14
no critic was trained in this stage and no K was chosen from noisy point estimates. Running `ADVANTAGE_COMPOSITE_CRITIC_V3`
at an eligible K is the sanctioned next objective; I did not launch it (it is a separate critic-training stage, and the
harm-asymmetry means its target should be framed as harm-rejection-plus-improvement, worth specifying before running).

## Claims / non-claims
**Claims:** (1) The counterfactual return signal ΔG increases monotonically with hold horizon K, paired-CI significant at
every K≥2 (preregistered, deterministic ×2). (2) The effect is real but heavy-tailed and predominantly destabilizing
(harmful > beneficial at every K; strict-success gain negative). (3) An eligible K for V3 exists under the frozen gate.
**Non-claims:** NOT that holding residuals improves delivery (net strict-success gain is negative). NOT that a V3 critic
will authorize improving actor updates (it may only reject harmful ones) — untested. NOT a sealed/actor result. Per-group
CIs are wide (10 groups/family, heavy tail); the monotone-increase claim rests on the paired gate, not the point medians.

## Next narrow experiment
`ADVANTAGE_COMPOSITE_CRITIC_V3` at K=16 (or the smallest passing K=2 for tightest control): regress the K-hold ΔG
within `state_group_id`, held out; report separately whether it (a) rejects harmful residuals and (b) ranks the minority
beneficial ones. If only (a), the residual line is a safety filter, not a policy-improvement lever.

## Files
- impl (commit A `62f0412`): `hymeko_rl/coin_delivery/coin_residual_hold_sweep.py`, `coin_counterfactual_labels.py`
  (gate-FSM snapshot + `label` flag), `experiments/…/coin_hold_sweep_v1.py`, `hymeko_rl/tests/test_coin_residual_hold_sweep.py`
- frozen config (commit B `8d85923`): `experiments/…/hold_sweep_v1_config.json`
- results (commit C): `experiments/…/hold_sweep_v1_results.json`, this report + `.json`,
  `reports/figures/2026-07-23-hold-sweep.png`, `experiments/…/make_hold_sweep_figures.py`

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Mac (Apple Silicon), torch 2.12.0, mujoco 3.10.0.
Fully deterministic (no seeds needed beyond the frozen isotropic-direction seed 20260723); ×2 identity certified.
