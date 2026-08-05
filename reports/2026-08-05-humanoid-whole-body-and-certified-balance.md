# A/2 whole-body coordination (negative) + C certified balance (positive) on the embodied humanoid

**Date:** 2026-08-05 · **Worktree:** `hymeko_humanoid` (head at start `3bf58f54`) · directions **A/2** and **C**.

## A/2 — whole-body coordination does not extend the range either
Following "not just the arms — the whole system available for coordination", I learned (CEM) a **whole-body linear
residual** over the certified `a=0` scaffold — a per-joint gain matrix `G` (16 joints × [pitch, pitch-rate,
CoM-offset]) — and evaluated held-out at the baseline's failure perturbation (4 rad/s).

**Result: Δ = +0.00, held-out AND on train.** The learned whole-body residual could not even overfit the failing
seeds. The certified scaffold is already near-optimal for **in-place** balance (ankle + hip + trunk + arm), whose
authority is capped at ~3 rad/s. The missing capability is **stepping** — moving a foot to the capture point —
which the bounded position-servo (a ±0.4 rad offset that holds `q0`, feet planted) cannot do. The reduced model
(a fixed-base torso) omitted stepping entirely, so it has nothing to transfer there. Honest: **control does not
transfer beyond the in-place limit; stepping is the real frontier and needs a different action space.**

## C — the VERIFICATION transfers cleanly
The verification arc (viability boundary + HSTL runtime monitor) applied to the real humanoid does carry over.
`humanoid_certified_balance.py`:
- **Viability boundary:** bisection on the certified scaffold's recoverable pitch-rate → **3.25 rad/s** (the
  certified recoverable region).
- **HSTL runtime monitor** over `G(safety_margin ≥ 0)` with `safety_margin = min(uprightness − 0.6, pelvis_z −
  0.55)` — BOTH failure modes (tipping and pelvis collapse; a uprightness-only spec missed the pelvis-collapse
  fall and read falsely safe). The same `make_monitor` / robust-STL used on the reduced model.

| perturbation | outcome | monitor robustness | satisfied | early warning |
|---|---|---|---|---|
| 2.0 rad/s | recovers | **+0.20** | yes | — |
| 4.5 rad/s (falls) | falls @199 | **−0.00** (< 0) | no | **warn @154 → lead 45 steps (45 ms)** |
| 4.5 rad/s (recovers) | recovers | +0.23 | yes | — |

So on the embodied humanoid the monitor **certifies recovery** (robustness > 0), **flags falls** (robustness < 0,
unsatisfied), and **warns ~45 ms before** the fall — a runtime safety signal on the real robot.

## The honest through-line
The reduced model's **control gains** (arm / whole-body in-place) do **not** transfer — the embodiment's inertia
ratios and the missing stepping DOF break them. The reduced model's **verification** (viability boundary + HSTL
monitor) **does** transfer, and gives a real runtime safety margin + early warning on the humanoid. Verify what you
can't yet control.

## Files
`scenarios/humanoid/humanoid_certified_balance.py` (+75), `tests/test_humanoid_certified_balance.py` (+35, 3
tests, `importorskip` mujoco), this report. No dependency change, no §1.

## Tests
`pytest tests/test_humanoid_certified_balance.py` → 3 passed (recoverable region; monitor certifies recovery;
monitor flags a fall with early warning); ruff clean.

## Follow-up
- **Stepping action space** (capture-point foot placement) — the frontier A/2 identified; a torque or CoP action
  space, not the bounded hold-`q0` servo.
- **Fit a learned viability boundary** (M0) on the humanoid's 2-D (pitch, pitch-rate) recoverable region and
  compare to the reduced model's; run the **Rust HSTL backend** for real-time monitoring.

## Provenance
Git SHA at start `3bf58f54`. Env: `.venv` (Python 3.11, MuJoCo, NumPy 2) + built `hymeko` CLI, macOS. Seeded.
