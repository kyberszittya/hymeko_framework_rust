"""Regression tests for the centroidal running trajectory optimizer — each pins a bug caught during the build.

Every test here would FAIL against a specific buggy version (per CLAUDE.md §3):
- dynamics feasibility  → the whole point of the optimizer (a plan that doesn't obey the CoM dynamics is junk);
- BOUNDED phase times   → the first-cut linear flight reward drove t_flight to 321 s (unbounded objective);
- ballistic flight      → flight knots must carry ZERO contact force (else it is not a real flight phase);
- friction cone         → stance forces outside |Fx| ≤ μ·Fz are not realisable on the ground;
- target speed          → the optimizer must actually hit the commanded run speed, not a slow shuffle.
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.centroidal_run import (
    CentroidalRunConfig,
    _dynamics_residual,
    _Packer,
    solve_run,
)


def _repack(cfg: CentroidalRunConfig, tr) -> np.ndarray:
    return np.concatenate([np.column_stack([tr.com, tr.vel]).ravel(),
                           tr.force[: cfg.ns].ravel(),
                           [tr.t_stance, tr.t_flight, tr.stride, tr.foot_x]])


def test_plan_satisfies_the_centroidal_dynamics() -> None:
    """The core contract: the CoM trajectory obeys m·CoM̈ = ΣF − mg to collocation tolerance."""
    cfg = CentroidalRunConfig(target_speed=1.2)
    tr = solve_run(cfg)
    resid = _dynamics_residual(cfg, _Packer(cfg.ns, cfg.nf), _repack(cfg, tr))
    assert np.abs(resid).max() < 1e-5                        # trapezoidal defect ~ 1e-9 in practice


def test_phase_times_are_bounded() -> None:
    """Regression: the first-cut linear flight reward made the objective UNBOUNDED (t_flight → 321 s).

    Bounds on the phase durations (and the apex-rise reward replacing the duration reward) keep the plan
    physical. This asserts a run stride is on human timescales, which the buggy version violated by ~10⁴×."""
    tr = solve_run(CentroidalRunConfig(target_speed=1.2))
    assert 0.0 < tr.t_flight <= 1.0                          # a ballistic hop, not a launch to orbit
    assert 0.05 <= tr.t_stance <= 1.5
    assert 0.0 < tr.stride < 2.0                             # a stride, not a teleport


def test_flight_phase_is_ballistic() -> None:
    """A real flight phase: the non-contact (flight) knots carry ZERO contact force, and it lasts ≥ min_flight."""
    cfg = CentroidalRunConfig(target_speed=1.5, min_flight=0.08)
    tr = solve_run(cfg)
    assert np.allclose(tr.force[~tr.contact], 0.0)           # no force while airborne
    assert tr.force[tr.contact, 1].min() >= cfg.fz_min - 1e-6    # feet loaded during stance
    assert tr.t_flight >= cfg.min_flight - 1e-6              # both feet off for a real interval


def test_stance_forces_respect_the_friction_cone() -> None:
    """Ground reaction is realisable: Fz ≥ 0 and |Fx| ≤ μ·Fz on every stance knot."""
    cfg = CentroidalRunConfig(target_speed=1.5)
    tr = solve_run(cfg)
    fx, fz = tr.force[tr.contact, 0], tr.force[tr.contact, 1]
    assert (fz > 0).all()
    assert (np.abs(fx) <= cfg.mu * fz + 1e-6).all()


def test_optimizer_hits_the_target_run_speed() -> None:
    """It produces a real RUN, not a shuffle: the achieved forward speed tracks the command (the CEM capped ~0.07)."""
    for target in (1.0, 1.5, 2.0):
        tr = solve_run(CentroidalRunConfig(target_speed=target))
        assert abs(tr.speed - target) < 0.15
        assert tr.speed > 0.8                                # decisively faster than the 0.07 m/s CEM hop
