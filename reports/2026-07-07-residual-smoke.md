# Residual + phase-gated smoke — FAIL, and it isolates the obstruction to the critic's GRADIENT

**Date:** 2026-07-07 · Git SHA `03b01c3` (working tree dirty). Non-core. One seed, CPU. Full safety stack PASS.
No SAC / plain actor-critic / multi-seed; v2b reward unchanged; monitor external. This is the residual branch the
CQL-smoke failure pointed to; it FAILED acceptance — but it is the deepest, cleanest result of the RL thread.

## Result — every mechanical failure mode eliminated, and it STILL degraded

| | ft_dom | monitor_pass | monitor_score | reward | gate_frac | violation |
|---|---:|---:|---:|---:|---:|---|
| baseline (this DAgger ckpt) | **0.75** | 0.417 | 0.278 | −115.2 | 0.28 | no_right_fingertip_contact |
| after residual (ε=0.03) | **0.542** | 0.167 | 0.143 | −133.6 | 0.13 | fingertips_never_approached |

**Acceptance: FAIL.** Aborts: `ft_dom_collapse` (0.75→0.54, −0.21), `monitor_score_drop`, `exploit_or_assisted_up`
(assisted 0.042→0.083). But note: **monitor_score stayed positive (0.143)** — this is a *degradation on the
manifold*, not the collapse-to-negative of the prior two smokes.

## What held — the structural safeguards all worked exactly as designed

- **STRONG_PASS critic, and it HELD through the run**: pre margins exploit 12.14 / ood_gap 15.13; post Q(dagger)
  **−12.5 ≫ Q(exploit) −24.2** (margin ~12). `q_not_inverted = True` on the key axis.
- **No Q-scale runaway**: Q(dagger) −11.9 → −12.5 (moved 0.65). The frozen critic structurally removed the runaway
  that drove the CQL actor smoke to −182. `no_q_runaway = True`.
- **No whole-policy drift**: `residual_norm_approach = 0.0` — the phase gate held the residual at *exactly zero* in
  APPROACH; the correction was confined to contact steps. `whole_policy_drift = False`, `gate_open_in_approach = False`.
- **Residual bounded**: training residual norm ~0.009, saturation 0.58 (< 0.9 abort), eval contact-norm 0.036
  (4-dim vector, per-dim ≤ ε=0.03). `residual_bounded = True`.
- **init ≡ DAgger** exactly; **provenance + tensor-contract PASS**.

## The finding — ranking-correct ≠ gradient-correct

The residual was trained to climb the frozen STRONG_PASS critic's Q, gated to contact, bounded to ε. It moved the
contact-phase action a tiny amount toward higher Q — and delivery got **worse** (ft_dom 0.75→0.54; contact
engagement *fell*, gate fraction 0.28→0.13, violation shifting to `fingertips_never_approached`). So:

> **The critic's local Q-gradient ∂Q/∂a is not aligned with the monitor, even where its ranking is STRONG_PASS-correct.**

A critic can order whole policies/actions correctly (dagger ≫ exploit by 12) while its *local improvement
direction* at DAgger states points slightly the wrong way. Following it with even a bounded, phase-gated,
drift-proof, runaway-proof residual degrades the policy. Ranking correctness (a global, ordinal property) does not
imply gradient correctness (a local, directional property) — and RL improvement rides on the gradient.

**Measured vs inferred vs hypothesis** (kept distinct per the operating contract):
- **Measured:** bounded gated residual vs STRONG_PASS frozen critic → ft_dom 0.75→0.54, monitor 0.278→0.143,
  on-manifold (no drift / runaway / negative collapse), residual ≤ ε contact-only.
- **Inferred:** the frozen critic's local gradient is not a trustworthy improvement direction on this task; the
  obstruction is neither ranking (fixed, held) nor actor drift (structurally capped) but the value *geometry*.
- **Hypothesis (needs a test):** the gradient *sign* is wrong, not merely its magnitude — no ε > 0 beats baseline.
  **Discriminating test (recommended, not run):** an ε-sweep {0.0, 0.01, 0.02, 0.03}; if ft_dom is monotone
  non-increasing in ε with no ε>0 above baseline, the direction itself is unhelpful (vs "ε merely too large").

## The three-attempt arc — each removed the prior failure and exposed the next-deeper one

1. **Baseline CTDE-TD3+BC** (guarded sanity) — failed via critic **mis-ranking** (Q(exploit) > Q(dagger)).
2. **CQL actor smoke** — fixed + held the ranking; failed via **Q-scale runaway + off-manifold drift**.
3. **Residual + phase-gated** (this) — fixed ranking + runaway + drift; failed via **critic gradient not improving**.

Each structurally-distinct attempt eliminated the previous mechanism and revealed the next. The residual smoke
removed every *mechanical* failure mode and exposed the *fundamental* one: on this task, the off-policy critic's
local gradient does not encode a valid improvement direction for the DAgger policy.

## Verdict + implication

**RL stays frozen. The imitation baseline (MLP+DAgger, ft_dom 0.452 deployable / 0.75 this checkpoint) stands.**
This is a strong, well-instrumented negative: the coin-delivery DAgger policy is **not improvable by off-policy
critic-gradient RL** even with (a) a ranking-correct conservative critic, (b) a frozen critic (no runaway), and
(c) a bounded phase-gated residual (no drift). The lever is not more actor-critic tuning.

If the line is pursued further (research, deliberate, gated), the options are — in order of how much they depart
from "trust the critic gradient":
- **ε-sweep** first (the discriminating test above) to confirm the gradient sign is the obstruction.
- **Gradient-accurate value learning** (not just ranking): a critic whose local slope matches the monitor — much
  harder; would need value accuracy on the DAgger manifold, not just correct ordering.
- **Gradient-free / monitor-directed improvement** (e.g. CEM/ES on the bounded residual scored by the monitor, or
  reconsidering — carefully, and only with your sign-off — the monitor as an in-the-loop signal, which the current
  contract forbids). This smoke is precisely the evidence that would justify revisiting that boundary.
- **Better imitation** (more/better demonstrations, DAgger rounds) rather than RL — consistent with the whole
  project finding that the lever past a BC/DAgger ceiling has been imitation, not off-policy RL.

## Files / artifacts

- `hymeko_rl/agents/residual_actor.py` (87 LOC) — `ResidualActor`, `build_residual_net`, `contact_gate`;
  `hymeko_rl/train/critic_repair.py` — `train_residual` (frozen critic, residual-only); tests
  `hymeko_rl/tests/test_residual_actor.py` (7 pass). Harness `scratchpad/v2_residual_smoke.py`; result
  `experiments/v2_residual_smoke/results.json`; residual net `experiments/v2_residual_smoke/residual_net_s1.pt`;
  log `scratchpad/residual_smoke.log`. All new code tested, ruff clean, CORE.YAML untouched, no new deps.

**Status:** residual + phase-gated smoke ran under the full safety stack and FAILED — but as an *on-manifold
degradation*, isolating the RL obstruction to the critic's local gradient. Three distinct RL attempts have now
each failed for a distinct, understood reason; the imitation baseline remains the deployable policy.
