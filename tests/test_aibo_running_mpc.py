"""Periodic running gait MPC — a centroidal limit cycle with a ballistic flight per stride."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from scenarios.aibo.hop_mpc import HopParams  # noqa: E402
from scenarios.aibo.running_mpc import RunningGaitMPC  # noqa: E402


def _sim(params: HopParams, v=0.6):
    mpc = RunningGaitMPC(p=params, v_forward=v)
    traj, sched, forces = mpc.simulate(n_strides=5)
    return mpc, traj, sched, forces


def test_running_advances_at_steady_forward_speed() -> None:
    mpc, traj, _s, _f = _sim(HopParams(mass=2.0, z0=0.23, f_max=80))
    assert traj[-1, 0] > 1.0                           # covers > 1 m over 5 strides
    assert abs(traj[:, 1].mean() - mpc.v_forward) < 0.1  # steady forward speed ~ target


def test_running_has_a_real_flight_phase() -> None:
    _m, _t, sched, _f = _sim(HopParams(mass=2.0, z0=0.23, f_max=80))
    assert (~sched).mean() > 0.3                       # substantial airborne fraction = a real run (not a walk)


def test_running_vertical_bounce_is_periodic() -> None:
    mpc, traj, _s, _f = _sim(HopParams(mass=2.0, z0=0.23, f_max=80))
    assert traj[:, 2].max() > mpc.p.z0 + 0.02          # rises above standing (apex)
    assert traj[:, 2].min() < mpc.p.z0 - 0.02          # crouches below (touchdown)


def test_capturability_stays_bounded_over_the_run() -> None:
    mpc, traj, _s, _f = _sim(HopParams(mass=2.0, z0=0.23, f_max=80))
    assert mpc.capture_lyapunov(traj).max() < 0.1      # recoverable every stride (orbital stability)


def test_running_forces_are_friction_feasible() -> None:
    _m, _t, _s, f = _sim(HopParams(mass=2.0, z0=0.23, mu=0.9, f_max=80))
    assert all(abs(f[k, 0]) <= 0.9 * f[k, 1] + 1e-4 for k in range(len(f)))
    assert f[:, 1].min() >= -1e-6                      # unilateral (Fz >= 0)


def test_human_runs_too() -> None:
    mpc, traj, sched, _f = _sim(HopParams(mass=15.0, z0=0.645, f_max=500))
    assert traj[-1, 0] > 1.0 and (~sched).mean() > 0.3 and mpc.capture_lyapunov(traj).max() < 0.1
