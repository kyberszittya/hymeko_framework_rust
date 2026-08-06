"""Contact-scheduled centroidal hop MPC — a PLANNED ballistic flight inside the capturability region."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scenarios.aibo.hop_mpc import CentroidalHopMPC, HopParams  # noqa: E402


def _plan(params: HopParams):
    mpc = CentroidalHopMPC(p=params, x_target=0.25)
    traj, f, sched = mpc.plan()
    return mpc, traj, f, sched


def test_hop_reaches_target_at_rest() -> None:
    mpc, traj, _f, _s = _plan(HopParams(mass=2.0, z0=0.23, f_max=60))
    xf = traj[-1]
    assert abs(xf[0] - mpc.x_target) < 0.03            # lands at the forward target
    assert abs(xf[1]) < 0.1 and abs(xf[3]) < 0.2       # at rest (vx, vz ~ 0)
    assert abs(xf[2] - mpc.p.z0) < 0.03                # back to standing height


def test_flight_phase_is_ballistic() -> None:
    _m, _t, f, sched = _plan(HopParams(mass=2.0, z0=0.23, f_max=60))
    assert np.max(np.abs(f[~sched])) < 1e-6            # zero ground force during flight (true ballistic)


def test_com_actually_leaves_the_ground() -> None:
    mpc, traj, _f, _s = _plan(HopParams(mass=2.0, z0=0.23, f_max=60))
    assert traj[:, 2].max() > mpc.p.z0 + 0.05          # the COM rises above standing -> a real hop (flight)


def test_capturability_lyapunov_stays_bounded() -> None:
    # the PLANNED loss of static stability stays within the recoverable Lyapunov region (not a fall)
    mpc, traj, _f, _s = _plan(HopParams(mass=2.0, z0=0.23, f_max=60))
    v = mpc.capture_lyapunov(traj)
    assert v.max() < 0.1                               # bounded throughout the flight
    assert v[-1] < 0.02                                # recovered at landing (capture point at the foot)


def test_forces_are_friction_feasible_and_bounded() -> None:
    _m, _t, f, sched = _plan(HopParams(mass=2.0, z0=0.23, mu=0.9, f_max=60))
    st = np.where(sched)[0]
    assert all(abs(f[k, 0]) <= 0.9 * f[k, 1] + 1e-5 for k in st)   # friction cone |Fx| <= mu Fz
    assert f[st, 1].min() >= -1e-6 and f[st, 1].max() <= 60 + 1e-6  # Fz in [0, f_max]


def test_human_params_also_plan_a_valid_hop() -> None:
    mpc, traj, f, sched = _plan(HopParams(mass=15.0, z0=0.645, f_max=400))
    assert abs(traj[-1, 0] - mpc.x_target) < 0.05 and traj[:, 2].max() > mpc.p.z0 + 0.05
    assert np.max(np.abs(f[~sched])) < 1e-6            # same ballistic flight for the human embodiment
