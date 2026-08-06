"""Pontryagin (PMP) optimal balance recovery — Hamiltonian structure + applied to the humanoid."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv  # noqa: E402
from scenarios.humanoid.lyapunov import evaluate_lyapunov  # noqa: E402
from scenarios.humanoid.pmp_recovery import PMPBalanceRecovery, riccati_from_hamiltonian  # noqa: E402


def test_hamiltonian_matrix_riccati_matches_scipy() -> None:
    # the PMP route (stable eigenspace of the Hamiltonian matrix) equals the algebraic Riccati solver
    sp = pytest.importorskip("scipy.linalg")
    p = PMPBalanceRecovery()
    P_ham = riccati_from_hamiltonian(p._A, p._B, p._Q, p.r)
    P_care = sp.solve_continuous_are(p._A, p._B, p._Q, np.array([[p.r]]))
    assert np.max(np.abs(P_ham - P_care)) < 1e-6


def test_lipm_optimal_recovery_decays() -> None:
    p = PMPBalanceRecovery()
    x = np.array([0.15, 0.0])
    for _ in range(400):
        x = x + 0.005 * (p._A @ x + p._B.flatten() * p.optimal_force(*x))
    assert abs(x[0]) < 0.02 and abs(x[1]) < 0.05           # COM offset + velocity -> 0 (optimal recovery)


def test_control_hamiltonian_near_zero_on_optimal() -> None:
    # PMP/HJB stationarity: H(x, Px, f*) ~ 0 along the infinite-horizon optimal manifold
    p = PMPBalanceRecovery()
    for off in (0.05, 0.1, 0.2):
        assert abs(p.control_hamiltonian(off, 0.0)) < 1e-3


def _cert(pitch: float):
    env = HumanoidBalanceEnv(cfg=BalanceConfig(max_steps=500), seed=0)
    mj, m, d = env._mj, env.model, env.data
    pmp = PMPBalanceRecovery()
    d.qpos[:] = env._q0
    d.qpos[env._base + 2] = 0.80
    d.qvel[:] = 0.0
    d.qvel[4] = pitch
    mj.mj_forward(m, d)
    vs, up, mjv = [], 0, 0.0
    for k in range(500):
        d.ctrl[:] = pmp.torque(env)
        mj.mj_step(m, d)
        vs.append(env.V(env._com_sig()))
        sig = env._com_sig()
        if sig["uprightness"] > 0.6 and float(d.xpos[env._pelvis, 2]) > 0.55:
            up = k + 1
        mjv = max(mjv, float(np.max(np.abs(np.asarray(d.qvel)[env._act_dof]))))
    return up, evaluate_lyapunov(vs)["passes"], mjv


def test_pmp_certifies_and_is_realistic() -> None:
    up, cert, mjv = _cert(0.3)
    assert up >= 495 and cert and mjv < 5.0               # certifies a 0.3 pitch at realistic joint speeds
