# PMP optimal balance recovery — Pontryagin's Maximum Principle on the underactuated LIPM

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION. Model-based (no RL).** · **Verdict: `PMP_OPTIMAL_RECOVERY_VIA_HAMILTONIAN_MATRIX`.**

---

## Why (the Hamiltonian optimum, not just stabilization)

Energy shaping *stabilizes*; PMP gives the *optimal* recovery. The sagittal balance is a Linear
Inverted Pendulum (LIPM) — the underactuated inverted pendulum at the heart of the floating
humanoid. For state `x = [com_off, com_vel]`, dynamics `ẋ = Ax + Bf` (`f` = horizontal COM
force), cost `J = ½∫(xᵀQx + rf²)dt`, **Pontryagin's Maximum Principle** gives the control
Hamiltonian and its necessary conditions:

    H(x,λ,f) = ½(xᵀQx + rf²) + λᵀ(Ax+Bf)
    ẋ = ∂H/∂λ,   λ̇ = −∂H/∂x,   ∂H/∂f = 0 ⟹ f* = −r⁻¹Bᵀλ

Eliminating `f` gives the **Hamiltonian (symplectic) system** `d/dt[x;λ] = M[x;λ]` with

    M = [[A, −B r⁻¹Bᵀ], [−Q, −Aᵀ]]     (the 2n×2n Hamiltonian matrix)

whose **stable eigenspace** yields `λ = P x` (the Riccati `P`) → `f* = −r⁻¹BᵀP x`.

## What we built (`scenarios/humanoid/pmp_recovery.py`)

- `hamiltonian_matrix` + `riccati_from_hamiltonian` — build `P` from the Hamiltonian matrix's
  **stable eigenspace** directly (the Pontryagin route), *not* a Riccati solver — to expose the
  PMP structure. **Verified `P` = `scipy.solve_continuous_are` to 0.0** (max abs diff).
- `PMPBalanceRecovery.optimal_force` — the PMP-optimal COM force `f* = −Kx` for the LIPM.
- `control_hamiltonian` — `H(x, Px, f*)`, **≈ 0 along the optimal manifold** (< 1e-3), the
  PMP/HJB stationarity hallmark.
- `torque(env)` — applies the optimal COM force to the full humanoid via the COM Jacobian
  (`τ = J_comᵀ f* + gravity + posture + damping`).

## Result

| check | outcome |
|---|---|
| Hamiltonian-matrix `P` vs `scipy` CARE | **0.0** (exact) |
| LIPM optimal recovery (com_off 0.15) | → **0.0025** (decays optimally) |
| control Hamiltonian on the optimal | **≈ 0** (< 1e-3) — HJB stationarity |
| applied to humanoid, pitch 0.2 / 0.3 | ✅ / ✅ certifies (0.8 / 1.0 rad/s) |
| pitch ≥ 0.4 | ❌ (same physical limit as the shaped controllers) |

The PMP-optimal feedback certifies the recovery to pitch 0.3 (matching the full-IDA-PBC basin),
at realistic joint speeds — and it is the **minimum-cost** recovery, with the control Hamiltonian
vanishing along the optimal trajectory.

## Files

```
scenarios/humanoid/pmp_recovery.py     NEW  (hamiltonian_matrix, riccati_from_hamiltonian, PMPBalanceRecovery)
tests/test_humanoid_pmp.py             NEW  4 tests (Riccati==scipy, LIPM decays, H≈0 on optimal, certifies+realistic)
```

## Tests / lint

`ruff` clean. **33/33 humanoid tests pass**, including 4 PMP tests locking: the Hamiltonian-matrix
route equals the Riccati solver, the LIPM recovery decays, the control Hamiltonian ≈ 0 on the
optimal, and the humanoid certifies a 0.3 pitch at realistic speeds.

## Bottom line

Pontryagin's Maximum Principle, built from the **Hamiltonian matrix's stable eigenspace**, gives
the **optimal** underactuated balance recovery (min-cost, control-Hamiltonian ≈ 0 on the optimal),
applied to the humanoid via the COM Jacobian — certifying to pitch 0.3 at realistic joint speeds.
This completes the Hamiltonian arc: energy shaping (stabilize) → full IDA-PBC (kinetic shaping,
larger basin) → PMP (optimal). The reduced LIPM/PMP core is **embodiment-agnostic** — it transfers
to the AIBO by the same optimal-force-through-the-COM-Jacobian mapping.
