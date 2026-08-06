"""Whole-body controller — contact-consistent inverse dynamics: stands, and transfers load off a foot.

Locks the validated WBC capabilities (the action-space change approved for walking): the KKT solve returns
finite torques; the WBC holds a stable stand (both-foot contact, CoM + orientation + posture tasks); and a
contact-force cost UNLOADS a foot in double support (the load-transfer primitive that the quasi-static
controllers could not achieve — the double-support wall in the walking-feasibility report). A full stable
gait built on top is a separate, in-progress prototype and is NOT asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv  # noqa: E402
from scenarios.humanoid.wbc import Task, WholeBodyController  # noqa: E402


def _env_wbc():
    e = HumanoidBalanceEnv(BalanceConfig(perturb_lo=0.0, perturb_hi=0.0, max_steps=4000))
    w = WholeBodyController(e.model, e.data, e._act_dof, "base")
    return e, w


def _settle(e, n=30):
    for _ in range(n):
        e.step(np.zeros(e.model.nu, np.float32))


def _foot_load(e, body: int) -> float:
    tot = 0.0
    for c in range(e.data.ncon):
        con = e.data.contact[c]
        if body in (e.model.geom_bodyid[con.geom1], e.model.geom_bodyid[con.geom2]):
            f = np.zeros(6)
            mujoco.mj_contactForce(e.model, e.data, c, f)
            tot += abs(f[0])
    return tot


def _stand_tasks(e, w, com0, pelR0, q0j):
    d = e.data
    jc = w.com_jacobian()
    com = np.asarray(d.subtree_com[1])
    comv = jc @ np.asarray(d.qvel)
    acc_com = 220.0 * (com0 - com) - 30.0 * comv
    _jp, jr = w.body_jacobian(e._pelvis)
    aa = w.orientation_error(d.xmat[e._pelvis].reshape(3, 3), pelR0)
    acc_pel = 140.0 * aa - 24.0 * (jr @ np.asarray(d.qvel))
    post = w.posture_task(q0j, e._act_qadr, kp=10.0, kd=5.0, weight=1.0)
    return [Task(jc, acc_com, 120.0), Task(jr, acc_pel, 45.0), post]


def test_solve_returns_finite_torques() -> None:
    e, w = _env_wbc()
    e.reset(seed=0)
    _settle(e)
    com0 = np.asarray(e.data.subtree_com[1]).copy()
    pelR0 = e.data.xmat[e._pelvis].reshape(3, 3).copy()
    tau = w.solve([e._fl, e._fr], _stand_tasks(e, w, com0, pelR0, e._q0j))
    assert tau.shape == (e.model.nu,) and np.all(np.isfinite(tau))


def test_wbc_holds_a_stable_stand() -> None:
    e, w = _env_wbc()
    e.reset(seed=0)
    _settle(e)
    com0 = np.asarray(e.data.subtree_com[1]).copy()
    pelR0 = e.data.xmat[e._pelvis].reshape(3, 3).copy()
    zs = []
    for _ in range(600):
        tau = w.solve([e._fl, e._fr], _stand_tasks(e, w, com0, pelR0, e._q0j))
        e.data.ctrl[:] = np.clip(tau, -150, 150)
        mujoco.mj_step(e.model, e.data)
        zs.append(float(e.data.xpos[e._pelvis][2]))
    zs = np.array(zs)
    assert e._com_sig()["uprightness"] > 0.9          # stays upright
    assert zs.std() < 0.01                             # pelvis height steady (no collapse/bounce)


def test_orientation_error_zero_on_identical() -> None:
    r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(WholeBodyController.orientation_error(r, r), 0.0, atol=1e-12)


def test_force_cost_unloads_a_foot_in_double_support() -> None:
    """The load-transfer primitive: a wrench cost on the swing foot drives its contact load toward zero
    while the WBC keeps balance — impossible for the quasi-static controllers (double-support wall)."""
    def run(wf: float) -> float:
        e, w = _env_wbc()
        e.reset(seed=0)
        _settle(e)
        com0 = np.asarray(e.data.subtree_com[1]).copy()
        pelR0 = e.data.xmat[e._pelvis].reshape(3, 3).copy()
        lxy = e.data.xpos[e._fl][:2].copy()
        loads = []
        for k in range(360):
            a = min(1.0, k / 150.0)
            jc = w.com_jacobian()
            com = np.asarray(e.data.subtree_com[1])
            comv = jc @ np.asarray(e.data.qvel)
            ctgt = com0.copy()
            ctgt[:2] = (1 - a) * com0[:2] + a * lxy
            acc_com = 260.0 * (ctgt - com) - 32.0 * comv
            _jp, jr = w.body_jacobian(e._pelvis)
            acc_pel = 140.0 * w.orientation_error(e.data.xmat[e._pelvis].reshape(3, 3), pelR0) \
                - 24.0 * (jr @ np.asarray(e.data.qvel))
            post = w.posture_task(e._q0j, e._act_qadr, kp=8.0, kd=4.0, weight=0.5)
            # contacts [left, right] -> right foot wrench is lambda[6:12]
            tau = w.solve([e._fl, e._fr], [Task(jc, acc_com, 150.0), Task(jr, acc_pel, 40.0), post],
                          force_cost=(slice(6, 12), wf * a))
            e.data.ctrl[:] = np.clip(tau, -150, 150)
            mujoco.mj_step(e.model, e.data)
            if k > 150:
                loads.append(_foot_load(e, e._fr))
        return float(np.min(loads))

    no_cost = run(0.0)
    with_cost = run(0.06)
    assert no_cost > 400.0                             # without the cost the swing foot stays heavily loaded
    assert with_cost < 120.0                           # the wrench cost transfers the load off it (unloads)
    assert with_cost < 0.3 * no_cost                   # a large, real reduction
