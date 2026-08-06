# Humanoid balance controller (option 2) + Lyapunov re-verification + SAC readiness

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov`
**SIMULATION. NOT RL (SAC assessed, not trained).**
**Verdict: `HANDTUNED_BALANCE_FAILS_LYAPUNOV_RL_OR_LQR_WARRANTED`.**

---

## Option 2 attempted — a floating-humanoid balance controller

Fixed two things from the first floating test: (1) **added a floor** (the earlier
floating humanoid had none → free fall), (2) the joints are all **sagittal y-hinges**
→ a 2D humanoid that tips only forward/backward, so an ankle strategy is 1D and
tractable. Tried two hand-tuned controllers, re-verified against the same Lyapunov
certificate:

| controller | Lyapunov | upright steps | V_final |
|---|---|---|---|
| fixed base, constrained (cannot fall) | ✅ | 3999 | 0.0 |
| floating+floor, gravity-comp PD (posture) | ❌ | 1169 | 1.82 |
| floating+floor, PD + ankle COM-feedback | ❌ | 1419 | 1.84 |

Both hand-tuned controllers **TIP (~1.2–1.4 s)** and **fail the Lyapunov certificate**.
The ankle strategy helps marginally (1169 → 1419 upright steps) but does **not**
stabilize. Root cause: the floating humanoid is an **underactuated inverted pendulum**;
gravity-comp PD holds joint *posture* and the ankle PD adds one feedback channel, but
neither closes the full-state loop needed to stabilize the unactuated base mode.

## Lyapunov certificate re-verified

The certificate **discriminates correctly**: it rejects both failing hand-tuned
floating controllers (V diverges) and passes only the constrained fixed base
(vacuously). Combined with AIBO (a real state-dependent controller that **passed**),
the same generic `lyapunov_certificate` now has a consistent, discriminating record
across four cases — a reward-independent stability verifier.

## SAC readiness assessment (feasibility, not trained)

The floating humanoid balance is a **genuine underactuated RL task** — hand tuning
fails, so it is not a shortcut-avoidable problem. It is **RL-ready** with the Lyapunov
machinery as the safety/metric backbone:

- **Task**: floating+floor humanoid falls (tips ~1.2 s) without a real balance loop.
- **Reward-independent gate**: the Lyapunov `V` is a natural cost (reward ∝ −V or
  −dV/dt), and `lyapunov_certificate` is the **reward-independent** success/safety
  certificate an RL run **must not change** — exactly the campaign's RL discipline
  (unchanged external certificate).
- **Regime (honest)**: there is **no certified hand-tuned baseline** (both failed), so
  SAC would be **from-scratch genuine RL** (like the coin R14–R60 line), NOT residual-
  over-scaffold. Alternatively, a **model-based LQR** (linearize about the standing
  equilibrium via `mjd_transitionFD` + Riccati) could yield a certified baseline, which
  would then enable **residual SAC** over it (the coin-R8 pattern that generalized).
- **Not run here**: SAC is a substantial training run; this is a feasibility +
  readiness assessment, not a trained result. Recommended order: LQR baseline first
  (a certified stable controller passing the Lyapunov certificate) → then residual SAC
  gated by the certificate; or SAC-from-scratch with the Lyapunov reward + certificate.

## Files (all NEW/scenario-side, non-core)

```
scenarios/humanoid/run_humanoid_balance.py   (floor + PD/ankle balance + Lyapunov re-verify + SAC assessment)
reports/2026-07-27-humanoid-lyapunov/balance_gates.json
```

## Tests + lint

`ruff` clean; the 10 humanoid tests (5 Lyapunov + 5 conformance) still pass.
CORE.YAML: none. Shared CIP core unchanged. NOT RL.

**Verdict:** a hand-tuned floating-humanoid balance controller (PD, ankle strategy)
**fails the Lyapunov certificate** — it tips at ~1.2–1.4 s. The certificate re-verifies
and discriminates. Balance here **warrants LQR or SAC**, with the reward-independent
Lyapunov certificate as the gate and V as the natural reward. SAC is feasible and the
task is genuine; the honest next step is an LQR certified baseline → residual SAC, or
SAC-from-scratch under the Lyapunov gate.
