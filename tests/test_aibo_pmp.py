"""PMP transferred to the AIBO — the reduced LIPM/Hamiltonian core works cross-embodiment, realistically."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv  # noqa: E402

from scenarios.aibo.capture_step import PushRecoveryLyapunov  # noqa: E402
from scenarios.aibo.lyapunov import evaluate_lyapunov  # noqa: E402
from scenarios.aibo.pmp_recovery import PMPQuadrupedRecovery, riccati_from_hamiltonian  # noqa: E402


def _env():
    return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12, max_steps=300)


def test_hamiltonian_riccati_matches_scipy() -> None:
    sp = pytest.importorskip("scipy.linalg")
    w2 = 9.81 / 0.23
    A = np.array([[0.0, 1.0], [w2, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([1.0, 0.1])
    P = riccati_from_hamiltonian(A, B, Q, 0.02)
    assert np.max(np.abs(P - sp.solve_continuous_are(A, B, Q, np.array([[0.02]])))) < 1e-6


def _rollout(vy: float):
    import mujoco
    env = _env()
    ctrl = PMPQuadrupedRecovery()
    paws = {k: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "paw_" + k) for k in ("fl", "fr", "bl", "br")}
    V = PushRecoveryLyapunov()
    env.reset(seed=0)
    env.data.qvel[1] = vy
    mujoco.mj_forward(env.model, env.data)
    z0 = {k: float(env.data.xpos[v, 2]) for k, v in paws.items()}
    vs, peak, air = [], 0.0, 0
    dofs = [int(env.model.jnt_dofadr[env.model.actuator_trnid[i, 0]]) for i in range(env.model.nu)]
    for _ in range(300):
        env.step(ctrl.action(env))
        vs.append(V(env))
        peak = max(peak, float(np.max(np.abs(np.asarray(env.data.qvel)[dofs]))))
        if all(float(env.data.xpos[v, 2]) - z0[k] > 0.03 for k, v in paws.items()):
            air += 1
    return vs, peak, air


def test_pmp_aibo_recovery_is_realistic_and_not_an_exploit() -> None:
    vs, peak, air = _rollout(0.8)
    assert evaluate_lyapunov(vs)["passes"]            # recovers a small push
    assert peak < 8.0                                 # realistic joint speeds (not the 27 rad/s exploit)
    assert air == 0                                   # feet stay down (postural, not the airborne sprawl)


def test_pmp_action_shape_and_bounds() -> None:
    env = _env()
    env.reset(seed=0)
    a = PMPQuadrupedRecovery().action(env)
    assert a.shape == (env.model.nu,) and np.all(np.abs(a) <= 1.0 + 1e-6)
