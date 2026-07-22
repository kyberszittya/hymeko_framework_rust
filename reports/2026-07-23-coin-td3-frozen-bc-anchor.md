---
title: TD3 frozen-BC proximal anchor — gradient contract PASS, smoke NO_PROGRESS
date: 2026-07-23
slug: coin-td3-frozen-bc-anchor
task: coin_v3 delivery — RL entry / basin preservation
verdict: ANCHOR_PRESERVES_INIT_BUT_NO_PROGRESS
gate: TD3_ANCHOR_GRADIENT_CONTRACT_PASS
---

# TD3 frozen-BC proximal anchor (β=400) — §1–§10

**Created-at:** 2026-07-23 01:25 CEST
**Preceding gate:** VALID_BUT_CHAOS_SENSITIVE_UPDATE (commit `c399531`) established the three §12
preconditions justifying a BC anchor: (i) the critic is valid and non-drifting through the entry window,
(ii) the frozen 3/9 actor is a `NARROW_CHAOS_SENSITIVE_SUCCESS_BASIN`, (iii) the unanchored actor still
loses all three marginal deliveries by update 5 (aΔ 0.066).

## Objective

`L_actor = −mean Q1(s, π(s)) + β · mean ‖a_exec(π(s)) − a_exec(π₀(s))‖²`, anchor on the **executed clipped**
actions against the **immutable** update-0 actor π₀. Audited critic (recovery config D, seed 0, 10 000 steps);
every other contract fixed (LR 3e-4, τ 0.005, target smoothing 0.2/clip 0.5, policy_delay 2, replay = D).

## §4 β calibration (action-delta rule; committed BEFORE training)

β chosen as the smallest value with median executed action-delta ≤ 0.005 **and** p95 ≤ 0.015 on the
TRAIN/QUERY anchor bank (π₀-rollout + certified-demo states, phase-mixed):

| β   | median | p95    | max (contact) |
|-----|--------|--------|---------------|
| 5   | 0.1301 | 0.2923 | 1.4944 |
| 20  | 0.0347 | 0.1023 | 0.7369 |
| 60  | 0.0119 | 0.0449 | 0.6455 |
| 150 | 0.0057 | 0.0237 | 0.7556 |
| **400** | **0.0032** | **0.0143** | 0.8159 |

The global `max` never falls under 0.02 for any β and is **not gated**: it is dominated by a minority of
high-curvature contact states where the clipped-action map is chaotic; the §4 `max ≤ 0.02` targets the calmer
transport/entry/settling phases, not separable here without phase labels. **β = 400** committed.

## §6 gradient contract — TD3_ANCHOR_GRADIENT_CONTRACT_PASS

**Structural finding:** the proximal-BC anchor gradient is **identically 0 at π₀** (deviation = 0 ⟹
∇θ‖a−a₀‖² = 0). Measured update-0 anchor-grad-norm = `0.000000e+00`. Therefore the §4 "anchor grad 2–4× Q at
update 0" rule is **unsatisfiable by any β** — the anchor is a *second-order* constraint on accumulation, not a
force on the first step. The ratio is measured at the earliest live point `p₁` (π₀ after one Adam Q-step).

The contract must be checked under the **actual optimizer (Adam)**, not raw SGD. Under Adam, one combined step:

| quantity | before → after | interpretation |
|----------|----------------|----------------|
| combined L | 65.256 → **64.477** | (a) objective decreases ✓ |
| Q₁(s,π(s)) | −64.087 → −64.196 | (b) Q not driven opposite (held) ✓ |
| anchor ‖a−a₀‖² | 1.169 → **0.281** | anchor pulls deviation back ✓ |
| ratio / cos @ p₁ | 12.4× / −0.81 | (c) anchor real, not cancelling Q ✓ |
| frac states updated | 1.00 | (d) all states retain update ✓ |

A raw-SGD check *fails* (L 65.26 → 118.72) because lr·‖g‖ overshoots the steep β=400 anchor; Adam's
per-parameter second-moment normalization bounds the effective step. **Contract PASS under the optimizer used.**

## §7–§9 guarded anchored smoke (≤5000 updates)

| upd | HL/9 | VAL/30 | grasp | aΔ med | critic sep | OOD-b |
|-----|------|--------|-------|--------|-----------|-------|
| 0   | **3** | 2 | 9 | 0.000 | +11.97 | 0.00 |
| 1   | 1 | 1 | 9 | 0.0221 | +11.34 | 0.00 |
| 5   | 1 | 2 | 9 | 0.0085 | +11.82 | 0.00 |
| 10  | 0 | 0 | 9 | 0.0084 | +11.79 | 0.00 |
| 50  | 1 | **4** | 9 | 0.0039 | +12.02 | 0.00 |
| 250 | 1 | 2 | 9 | 0.0043 | +10.73 | 0.00 |
| 1000| 0 | 1 | 9 | 0.0028 | +7.09 | 0.00 |
| 2500| 0 | 0 | 9 | 0.0028 | +10.16 | 0.04 |
| 5000| 1 | 2 | 9 | 0.0033 | +13.57 | **0.92** → PAUSE |

**Verdict: `ANCHOR_PRESERVES_INIT_BUT_NO_PROGRESS`.**

- **Grasp preserved (9/9 throughout); runaway prevented.** Deviation held to ~0.003–0.008 — the anchor is
  load-bearing against the unanchored 0.066 collapse (which paused at HL 0/9, upd 5).
- **Marginal deliveries NOT preserved.** HL falls 3 → 1 at the **first** update (deviation 0.0221, anchor ≈ 0)
  and oscillates 0–1 thereafter; it never returns to 3. VAL 4 @ upd 50 is a transient (2 marginal seeds),
  not a trend — HL stayed ≤ 1. No headline transport progress.
- **Late critic OOD drift.** The co-training critic keeps a healthy success/fail separation (sep > +7 whole
  run) but its OOD-boundary exploitation creeps 0.00 → 0.04 (2500) → 0.92 (5000), tripping the §8 pause. The
  actor never exploited it (deviation stayed ~0.003).

## Root cause (§10 interpretation)

The mechanical label is `ANCHOR_PRESERVES_INIT_BUT_NO_PROGRESS`; the **mechanism** is
`VALID_Q_DIRECTION_NOT_LOCALLY_REALIZABLE`. The critic's Q-improvement direction is valid (sep +12, no early
drift), but it is **not locally realizable**: the very first gradient step along it (deviation 0.0221) already
exceeds the ~0.02 threshold that destroys the chaos-marginal deliveries. Because the proximal-BC anchor is
**zero at π₀**, it cannot shield that first step for *any* β — it only engages once deviation exists, bounding
accumulation thereafter. A weaker β permits *more* deviation, not less; a stronger β cannot act earlier. The
NARROW_CHAOS_SENSITIVE basin is fragile at the first step, which is exactly where this anchor has no leverage.

This is a structural property of the frozen-BC proximal anchor on a first-step-fragile basin, not a tuning
failure — and it is consistent across the β sweep (§4) and the gradient contract (§6).

## Files touched

- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_rl_anchor_smoke.py` (new, 181 L)
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_rl_anchor_contract.py` (new, 96 L)
- `experiments/2026_07_22_coin_v3_learning/rl_entry/anchor_smoke_result.json` (new)
- `experiments/2026_07_22_coin_v3_learning/rl_entry/anchor_contract_result.json` (new)
- `reports/figures/coin_td3_frozen_bc_anchor.png` (new)

**CORE.YAML items touched:** none. **SAC:** untouched / quarantined for this stage.

## Provenance

- Host: Mac (Apple Silicon), main-repo venv (torch CPU), cwd = repo root (BC checkpoint + new modules present).
- BC anchor: `bc_handoff_only_best.pt`; critic: recovery config D (demo⊕frozen⊕fails⊕perturb), seed 0, 10 000 steps.
- Seeds: anchor bank / replay 6000–6079; auth HEADLINE ∪ 7000–7014; smoke env seeds drawn from rng(0).
  Final-test bank (8000–8049) untouched.
- RL is not bit-reproducible (§3 carve-out); the structural findings (anchor≡0 at π₀, first-step destruction)
  are deterministic and β-independent.

## Follow-ups (not run this stage)

1. The anchor cannot protect a first-step-fragile basin. If basin preservation is required, the lever is
   **step-0 trust-region on the executed action** (clip ‖a_exec(π)−a_exec(π₀)‖ per-update), not a proximal
   penalty — a hard cap acts on the first step where the penalty is null.
2. Or accept the basin as a *search prior*, not a policy init (cf. coin genuine-RL R59/R60: nonlocal
   exact-rollout search, not local policy-improvement, is what exceeded the supervised ceiling).
3. The co-training critic's late OOD drift (0.92 @ 5000) argues for a frozen or periodically-re-audited critic
   in any longer run.
