"""Floating-base HyMeKo humanoid balance env for SAC (gym 5-tuple).

The LQR path stalled on contact-consistent equilibrium + contact-mode-robust
linearization; SAC sidesteps both. The reward is driven by the COM Lyapunov energy
(reward = alive − w·V − control cost), and the reward-independent
``lyapunov_certificate`` is the success/safety gate (evaluated separately, never in
the reward). Gravity-comp (qfrc_bias) is a feedforward physics term (it does NOT
balance — it tips); SAC learns the balancing residual torque on top.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .lyapunov import HumanoidCOMLyapunov

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "data" / "robotics" / "humanoid.hymeko"


def _cli() -> Path:
    for prof in ("release", "debug"):
        p = _REPO / "target" / prof / "hymeko"
        if p.is_file():
            return p
    raise FileNotFoundError("hymeko CLI not built")


def _build():
    import mujoco
    xml = subprocess.run([str(_cli()), "emit", "-f", "mjcf", str(_SRC), "-n", "humanoid"],
                         capture_output=True, text=True, check=True).stdout
    xml = xml.replace('<joint name="base" type="hinge" axis="0 0 1"/>', '<freejoint name="base"/>')
    xml = xml.replace('<motor name="act_base" joint="base" gear="1"/>\n    ', '')  # unactuated base
    xml = xml.replace('<worldbody>',
                      '<worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1" '
                      'pos="0 0 0" condim="3" friction="1 0.1 0.1"/>')
    return mujoco, mujoco.MjModel.from_xml_string(xml)


class HumanoidBalanceEnv:
    """gym-5-tuple floating-humanoid balance env. Action = 12 normalised joint torques."""

    def __init__(self, max_steps: int = 500, torque: float = 50.0, seed: int = 0) -> None:
        self._mj, self.model = _build()
        self.data = self._mj.MjData(self.model)
        self._mj.mj_forward(self.model, self.data)
        self._q0 = self.data.qpos.copy()
        self._base = int(self.model.jnt_qposadr[self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, "base")])
        self._h_ref = 0.818
        self._pelvis = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "pelvis")
        self._fl = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "foot_l")
        self._fr = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "foot_r")
        self._act_dof = [int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]) for i in range(self.model.nu)]
        self._act_qadr = [int(self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]) for i in range(self.model.nu)]
        self.V = HumanoidCOMLyapunov(h_ref=self._h_ref)
        self.max_steps = max_steps
        self.torque = torque
        self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)

    def _com_sig(self) -> dict:
        com = self.data.subtree_com[1]
        support = 0.5 * (self.data.xpos[self._fl][:2] + self.data.xpos[self._fr][:2])
        return {"com_z": float(com[2]),
                "com_xy_off": float(np.linalg.norm(com[:2] - support)),
                "com_speed": float(np.linalg.norm(self.data.cvel[1][:3])),
                "uprightness": float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2])}

    def _obs(self) -> np.ndarray:
        m = self.data.xmat[self._pelvis].reshape(3, 3)
        com = self.data.subtree_com[1]
        support = 0.5 * (self.data.xpos[self._fl][:2] + self.data.xpos[self._fr][:2])
        jq = np.array([self.data.qpos[a] for a in self._act_qadr])
        jv = np.array([self.data.qvel[dof] for dof in self._act_dof])
        return np.concatenate([
            [m[2, 2], m[0, 2], float(self.data.xpos[self._pelvis, 2]) - self._h_ref],
            self.data.qvel[0:6], jq, jv,
            [float(com[0] - support[0]), float(self.data.cvel[1][0])],
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.data.qpos[:] = self._q0
        self.data.qpos[self._base + 2] = 0.80
        self.data.qvel[:] = 0.0
        self.data.qvel[4] = float(self._rng.uniform(-0.1, 0.1))   # small pitch-rate perturbation
        self._mj.mj_forward(self.model, self.data)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        tau = np.zeros(self.model.nu)
        for i, dof in enumerate(self._act_dof):
            tau[i] = a[i] * self.torque + float(self.data.qfrc_bias[dof])   # gravity-comp + SAC residual
        self.data.ctrl[:] = tau
        self._mj.mj_step(self.model, self.data)
        self._t += 1
        sig = self._com_sig()
        v = self.V(sig)
        upright = sig["uprightness"] > 0.6 and float(self.data.xpos[self._pelvis, 2]) > 0.55
        fell = (not upright) or not np.all(np.isfinite(self.data.qpos))
        reward = 1.0 - 2.0 * v - 0.001 * float(np.sum(a * a))       # alive - Lyapunov - control cost
        terminated = fell
        truncated = self._t >= self.max_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), {"V": v, "upright": upright}
