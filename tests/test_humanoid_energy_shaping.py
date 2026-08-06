"""Energy-shaping (IDA-PBC) balance — balances realistically, shaped energy is a Lyapunov fn."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv  # noqa: E402
from scenarios.humanoid.energy_shaping import EnergyShapingBalance, KineticShapedBalance  # noqa: E402
from scenarios.humanoid.lyapunov import evaluate_lyapunov  # noqa: E402


def _rollout(pitch: float, steps: int = 500, ctrl=None):
    env = HumanoidBalanceEnv(cfg=BalanceConfig(max_steps=steps), seed=0)
    mj, m, d = env._mj, env.model, env.data
    ctrl = ctrl or EnergyShapingBalance()
    d.qpos[:] = env._q0
    d.qpos[env._base + 2] = 0.80
    d.qvel[:] = 0.0
    d.qvel[4] = pitch
    mj.mj_forward(m, d)
    Hs, Vs, up, mjv = [], [], 0, 0.0
    energy = getattr(ctrl, "shaped_energy", lambda _e: 0.0)        # kinetic controller has no shaped_energy
    for k in range(steps):
        d.ctrl[:] = ctrl.torque(env)
        mj.mj_step(m, d)
        Hs.append(energy(env))
        Vs.append(env.V(env._com_sig()))
        sig = env._com_sig()
        if sig["uprightness"] > 0.6 and float(d.xpos[env._pelvis, 2]) > 0.55:
            up = k + 1
        mjv = max(mjv, float(np.max(np.abs(np.asarray(d.qvel)[env._act_dof]))))
    return env, ctrl, Hs, Vs, up, mjv


def test_energy_shaping_balances() -> None:
    _e, _c, _H, _V, up, _mjv = _rollout(0.2)
    assert up >= 495                                  # stays upright the whole episode


def test_shaped_energy_is_a_lyapunov_function() -> None:
    _e, _c, Hs, _V, _up, _mjv = _rollout(0.2)
    descent = sum(1 for a, b in zip(Hs, Hs[1:]) if b <= a + 1e-3) / (len(Hs) - 1)
    assert min(Hs) >= -1e-9                            # H_d >= 0 (positive definite)
    assert descent >= 0.85                             # near-monotone decrease (damping injection)


def test_energy_shaping_is_physically_realistic() -> None:
    _e, _c, _H, _V, _up, mjv = _rollout(0.3)
    assert mjv < 5.0                                   # realistic joint speeds (not the 27 rad/s exploit)


def test_energy_shaping_certifies_small_perturbation() -> None:
    _e, _c, _H, Vs, _up, _mjv = _rollout(0.15)
    assert evaluate_lyapunov(Vs)["passes"]             # COM certificate passes for a small perturbation


def test_com_jacobian_term_shape() -> None:
    env = HumanoidBalanceEnv(cfg=BalanceConfig(), seed=0)
    env.reset(seed=0)
    j = EnergyShapingBalance()._com_jac_xy(env)
    assert j.shape == (2, env.model.nu)               # d(com_xy)/d(q_actuated)


def test_kinetic_shaping_enlarges_certified_basin() -> None:
    # full IDA-PBC (task-space COM inertia = kinetic shaping) certifies pitch 0.3 where potential-only fails
    _e, _c, _H, v_pot, _u, _m = _rollout(0.3, ctrl=EnergyShapingBalance())
    _e2, _c2, _H2, v_kin, _u2, _m2 = _rollout(0.3, ctrl=KineticShapedBalance())
    assert not evaluate_lyapunov(v_pot)["passes"]     # potential-shaping fails a 0.3 pitch
    assert evaluate_lyapunov(v_kin)["passes"]         # kinetic shaping (larger basin) certifies it


def test_kinetic_shaping_realistic() -> None:
    _e, _c, _H, _V, up, mjv = _rollout(0.3, ctrl=KineticShapedBalance())
    assert up >= 495 and mjv < 5.0                    # balances + realistic joint speeds
