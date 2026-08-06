"""Closed-loop (receding-horizon) capturability MPC — rejects a mid-run push; open loop drifts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scenarios.aibo.closed_loop_mpc import ClosedLoopRunningMPC  # noqa: E402
from scenarios.aibo.hop_mpc import HopParams  # noqa: E402
from scenarios.aibo.running_mpc import RunningGaitMPC  # noqa: E402


def _mpc():
    return ClosedLoopRunningMPC(p=HopParams(mass=2.0, z0=0.23, f_max=80), v_forward=0.6)


def test_closed_loop_rejects_a_mid_run_push() -> None:
    mpc = _mpc()
    _t, _s, vxerr = mpc.simulate(n_strides=6, push_stride=3, push_dvx=0.4)
    cyc = mpc.n_stance + mpc.n_flight
    assert vxerr[3 * cyc:].max() > 0.2                 # the push perturbs the speed
    assert vxerr[-1] < 0.1                             # ...and the MPC settles back to the target speed


def test_capturability_bounded_under_the_push() -> None:
    mpc = _mpc()
    traj, sched, _v = mpc.simulate(n_strides=6, push_stride=3, push_dvx=0.4)
    assert mpc.capture_lyapunov(traj).max() < 0.15     # stays recoverable through the disturbance
    assert (~sched).mean() > 0.3                       # still a real run (flight phase preserved)


def test_open_loop_drifts_under_the_same_push() -> None:
    # contrast: applying the fixed nominal forces does NOT reject the push
    nom = RunningGaitMPC(p=HopParams(mass=2.0, z0=0.23, f_max=80), v_forward=0.6)
    f0, ztd, vztd = nom.plan_stride()
    prof = np.vstack([f0, np.zeros((nom.n_flight, 2))])
    x, m, g = np.array([0.0, 0.6, ztd, vztd]), 2.0, 9.81
    for s in range(6):
        if s == 3:
            x[1] += 0.4
        for fx, fz in prof:
            x = x + 0.02 * np.array([x[1], fx / m, x[3], fz / m - g])
    assert abs(x[1] - 0.6) > 0.3                        # open loop keeps the extra speed (no rejection)
