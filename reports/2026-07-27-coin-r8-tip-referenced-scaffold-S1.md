# R8 — tip-referenced scaffold: S1 safety de-risk (joints regulated; coin-fling is the residual RL's job)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r8-tip-referenced-residual-rl` (from R7 tip `81efd831`) · Contract:
`reports/2026-07-27-coin-r8-corrected-rl-gate-contract.md` (the corrected, non-circular RL gate).

**Result in one line:** regulating the **directly-actuated joint (tip) velocity** — R7's indicated fix — succeeds where R7's
coin-velocity servo failed: the **joints are now bounded and stable** (peak q̇ ≤ 2.03 < the 3.0 hard limit; anti-windup via
the slew clip; no NaN), so the scaffold is a valid, safe joint controller. But the **frictional grip is not rigid**: the coin
can be flung *ahead of* the bounded tips (s3 coin peak 1.57 m/s > the ee-speed limit 1.5, then reverses), so bounding the
joints does not by itself bound the coin. Under gentler gains the fling is brought within the motion contract (see §2). This
is exactly the residual RL's remit under the corrected gate: a small learned δ over this safe scaffold shaping the impulsive
tip→coin coupling that no fixed gain fully tames.

## 1. S1 v1 (v_max 0.30, q̇_max 2.0) — joints bounded, s3 coin flung over the ee limit

| state | coin peak | peak q̇ | min v∥ | dtz start→end | within limits |
|---|---|---|---|---|---|
| s1 | 0.36 | 1.66 | −0.21 | 76→48 mm | ✓ |
| s3 | **1.57** | 2.02 | −1.33 | 100→482 mm | ✗ (coin > 1.5) |
| s4 | 0.80 | 1.99 | −0.57 | 96→208 mm | ✓ |
| s7 | 1.11 | 2.03 | −0.85 | 138→266 mm | ✓ |

**Every state's joints stay bounded** (peak q̇ ≤ 2.03 < 3.0) — the R7 *joint* blow-up is gone; the joint-velocity servo is a
stable, anti-wound controller. But the coin, held by a compliant frictional grip, is not a rigid extension of the tips:
tip q̇ ≤ 2 rad/s (tip speed ≈ 0.3 m/s) yet the s3 coin reaches 1.57 m/s — a contact-impulse **fling**, then a reversal. So
bounding the joints ≠ bounding the coin; the tip→coin map is impulsive/compliant (the same soft-contact nonlinearity R7
localised, now isolated *at the joint→coin interface* with the joint side controlled).

## 2. S1 gentle-gain (safe-scaffold) — **S1 PASSES** at v_max 0.14 / q̇_max 1.0 / k_q 5.0 (frozen default)

| config | s1 | s3 | s4 | s7 | S1 all-safe |
|---|---|---|---|---|---|
| v_max 0.18, q̇_max 1.3 | ✓ (0.36) | ✓ (1.24) | ✗ (1.57) | ✓ (0.82) | ✗ |
| **v_max 0.14, q̇_max 1.0** | ✓ (0.36) | ✓ (1.11) | ✓ (1.20) | ✓ (0.67) | **✓** |

(coin peak in parentheses; ee limit 1.5.) At **v_max 0.14 / q̇_max 1.0** all four cradles are within the motion contract —
coin peak ≤ 1.20 < 1.5, joints ≤ 1.08 < 3.0, stable, anti-wound — so the tip-referenced scaffold is a **safe deterministic
scaffold (S1 PASS)**. It does **not** deliver (s3/s4 still overshoot/reverse to ~260 mm; s1/s7 reach 58/51 mm; residual
reversal −0.26…−0.87) — but delivery is **not** an S1 requirement; that overshoot/reversal is exactly the impulsive
tip→coin behaviour the bounded residual RL is to shape. Frozen scaffold default set to these gains.

## 3. Reading against the corrected gate

Per the corrected S1 (`…-gate-contract.md`): a *safe scaffold* needs bounded tip velocity, torque/slew limits, numerical
stability, anti-windup, and no unbounded energy — **not** K6 4/4. The joint-velocity servo delivers the bounded, stable,
anti-wound joint control S1 asks for. The coin-speed limit (ee ≤ 1.5) is a motion-contract constraint the scaffold must also
meet; the gentle-gain sweep (§2) is the dev-only tune that brings s3 within it. **The coin-fling / reversal is precisely the
nonlinear, cradle-specific, impulsive contact behaviour the bounded residual RL is meant to shape** — the teacher's own θ
proves the coin *is* controllable, so it is learnable over this safe base.

## 4. Status & next

Built: `tip_transport.py` (tip-referenced joint-velocity servo + anti-windup + squeeze decay + R6 certificate; cradle-
agnostic; reuses `velocity_ref`, the frozen force directions, `velocity_rollout`). 16 fast tests pass (incl. the R8 bounded-
reference test); ruff clean; dtau_for_step CC 11. **S1 = joint control safe/bounded; coin-fling within limits under the
gentle-gain tune (§2).** **SAC/TD3 still not started** — the corrected gate authorises the bounded residual RL only after
S1–S3; this session established the S1 scaffold. **Next phase (gated):** S2 update-zero adapter (δu = 0 reproduces this
scaffold to tolerance) → S3 learnability probe on dev → S4 matched SAC/TD3 on dev → S5 frozen held-out. Every §1
integrity constraint (no teleport/hidden-force/teacher-fallback, exact provenance, reward ⊥ K6, held-out excluded from
training/tuning, oracle 4/4) holds throughout.
