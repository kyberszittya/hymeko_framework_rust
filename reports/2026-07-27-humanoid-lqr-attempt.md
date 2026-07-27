# Humanoid LQR balance — attempt + layered diagnosis (honest negative)

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov`
**SIMULATION. NOT RL.**  ·  **Verdict: `LQR_NEEDS_A_CONTACT_CONSISTENT_EQUILIBRIUM_SOLVER` (focused controls build; not a quick win).**

---

## Goal

A model-based LQR certified baseline (linearize about the standing equilibrium via
`mjd_transitionFD` + discrete Riccati) that PASSES the Lyapunov certificate and
scaffolds residual SAC. I attempted it and hit three layered obstacles — two fixed,
one remaining — and stopped rather than spiral.

## Layered diagnosis

1. **Spurious base actuator (FIXED).** The emitted MJCF has `<motor name="act_base"
   joint="base" gear="1"/>` with **no ctrlrange**. After the freejoint edit this
   motor targets the base translation dof → a fake, unbounded "base actuator" that
   poisons `B` and gives `|K| ≈ 8.5e7`. Removed it via a string replace (`nu` 13→12);
   the base is now genuinely unactuated.

2. **Uncontrollable sagittal-plane modes → DARE unsolvable (FIXED by reduction).**
   All joints are y-hinges (sagittal), so the base **y-translation, roll, and yaw are
   uncontrollable and sit on the unit circle** — `solve_discrete_are` raises
   *"Failed to find a finite solution"* on the full 36-D state. Fix: reduce to the
   **controllable sagittal subspace** (keep base x, z, pitch + the 12 joints; drop
   y/roll/yaw) → the DARE then solves.

3. **Contact-consistent equilibrium (REMAINING BLOCKER).** LQR needs a true fixed
   point (`qacc ≈ 0` in *forward* dynamics at `(q_eq, 0, u_eq)`). Both quick routes
   failed: `mj_inverse` gives `u_eq` that leaves `eq_qacc ≈ 80` (mj_inverse ≠ inverse
   of mj_forward with contacts), and a PD-settle then read-back gives `eq_qacc ≈ 1.7e4`
   (the pose is mid-tip / feet penetrate). Without `qacc ≈ 0`, the linearization is
   about a non-equilibrium → `|K|` huge → the closed loop diverges (falls in ~0.3 s).

## Honest conclusion

A certified LQR balance baseline for this floating humanoid is a **focused controls
build**, not a quick smoke: it needs a **proper static-equilibrium solver** (a QP:
minimize ‖qacc‖ over `(q, u)` subject to the feet-contact and unactuated-base
constraints) **plus** the sagittal-subspace reduction (obstacle 2). Obstacles 1 and 2
are solved; obstacle 3 is the remaining work. This is real and worth doing, but
rushing the equilibrium solver risks a wrong result.

## Why SAC-from-scratch is the pragmatic alternative

SAC **sidesteps all three obstacles** — no equilibrium, no linearization, no
subspace reduction. It learns the balance policy directly, and the **Lyapunov
machinery already built is exactly the reward/gate it needs**:

- reward ∝ −V (or −dV/dt) — drive the COM energy down;
- the reward-independent `lyapunov_certificate` gates success/safety (unchanged
  external certificate — the campaign's RL discipline);
- no certified hand-tuned baseline exists → SAC-from-scratch (genuine RL, coin R14–R60
  regime), NOT residual-over-scaffold.

## Recommendation (two honest paths)

1. **LQR path** — implement the contact-consistent equilibrium QP + sagittal-subspace
   LQR (a focused controls session). Yields a *certified* baseline → residual SAC
   (coin-R8), the strongest positive if it lands.
2. **SAC path** — SAC-from-scratch under the Lyapunov reward + certificate gate.
   Sidesteps the controls subtleties; a larger training run.

## What is genuinely delivered

The floating-base HyMeKo humanoid (freejoint + floor + unactuated base) is a real,
reusable balance testbed; the three obstacles are precisely diagnosed; and the
Lyapunov certificate is ready as the RL reward/gate. No LQR baseline is claimed — it
is not working, and it is not committed as scenario code. NOT RL, core unchanged.
