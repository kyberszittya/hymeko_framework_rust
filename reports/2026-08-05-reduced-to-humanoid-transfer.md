# Reduced-model → embodied humanoid transfer: the arm reaction-wheel does NOT carry over

**Date:** 2026-08-05 · **Worktree:** `hymeko_humanoid` (head at start `479df0f7`) · direction **A**.

## Summary
Tested whether the reduced balance model's prediction — reaction-wheel arms extend the recoverable pitch basin
(+0.30, `reaction_wheel_arms.py`) — transfers to the real MuJoCo humanoid (`balance_env`, 16 actuated joints
incl. `shoulder_l/r`, `elbow_l/r`). `humanoid_transfer.py` runs a **bounded arm-swing residual over the certified
PD-hold scaffold** (shoulders swing ∝ torso pitch / pitch-rate) vs the `a=0` baseline, over a pitch-perturbation
sweep.

## Result — negative transfer, diagnosed
| pitch-rate perturb (rad/s) | baseline survive | + arm reaction-wheel | Δ |
|---|---|---|---|
| ≤ 2.0 | 1.00 | 1.00 | +0.00 |
| 3.0 | 1.00 | 1.00 | +0.00 |
| 4.0 – 6.0 | 0.25 | 0.25 | **+0.00** |
| 7.0 | 0.00 | 0.00 | +0.00 |

- The **certified `a=0` PD-hold scaffold is already robust** to pitch-rate perturbations up to ~3 rad/s — the
  transfer of the *certified-scaffold* idea works.
- Where the baseline fails (≥ 4 rad/s: the body pitches over and the pelvis collapses — measured: uprightness
  0.69, pelvis_z 0.55 at fall), the **arm residual adds nothing (Δ = 0.00)** across every gain/sign tried.
- **Diagnosis:** the arms' angular-momentum authority — small arm inertia, a ±0.4 rad target range — is far below
  a whole-body base rotation at 4 rad/s (`arm inertia << torso`). The reduced model's favourable arm/torso inertia
  ratio **over-predicted** the arms' value.

**Honest lesson:** a reduced-model gain must be re-checked against the embodiment's *actual inertia ratios* before
it is claimed — the abstraction that made the arm a strong reaction wheel does not hold on this humanoid. (This is
the same "measure, don't assume" discipline the balance-env doc already records for `h_ref` and the action model.)

## Files
`scenarios/humanoid/humanoid_transfer.py` (+70), `tests/test_humanoid_transfer.py` (+30, 2 tests, `importorskip`
mujoco), this report. No dependency change, no §1.

## Tests
`pytest tests/test_humanoid_transfer.py` → 2 passed (baseline robust to 2 rad/s; arm residual Δ≈0 in the failure
regime); ruff clean.

## Follow-up (honest, open)
- The negative is for a **base-pitch** perturbation exceeding arm authority; a **lateral push** (frontal plane) or
  a **near-edge** perturbation, or a true torque-reaction arm command (not a position-servo target), might differ —
  open, not refuted.
- What *does* transfer is the **certified scaffold** (a=0) and the capture-point step; the reduced model's value
  to the embodied humanoid is the certificate/coordination, not the arm gain.

## Provenance
Git SHA at start `479df0f7`. Env: `.venv` (Python 3.11, MuJoCo, NumPy 2) + built `hymeko` CLI, macOS. Seeded.
