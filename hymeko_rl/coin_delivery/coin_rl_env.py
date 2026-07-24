"""Canonical 4-DoF (inner.step arm) RL env for the SAC/TD3 entry campaign — the frozen handoff BC's action space,
with the EXACT canonical reward (`inner.reward_spec.evaluate` on the v3 spec, faithful to `CoinDeliveryTrainEnv._
transition`) and the canonical K=6 strict-dwell termination. obs = node_features flat 48; action = 4 arm actuators in
[-4, 4]; terminated on the 6th consecutive strict step (centered + settled + robot-attributed); truncated at horizon.

No dynamics/CORE/reward change — this wraps the existing neutral env + the loaded v3 RewardSpec so the RL consumes the
same reward the BC's own success is graded against.
"""
from __future__ import annotations

import numpy as np

CENTER_TOL = 0.02
SETTLE_VEL = 0.06
HELD_DWELL = 6
CANONICAL_REWARD_FILE = "data/robotics/galambos_task_deliver_v3.hymeko"


class CoinRL4Dof:
    """Gym-like 4-DoF canonical-reward env. `reset(seed)` → obs(48); `step(a4)` → (obs, reward, terminated, truncated)."""

    def __init__(self, horizon: int = 360, *, geom: str | None = None, arm_mjcf_transform=None):
        from hymeko_rl.env.reward import RewardSpec
        from hymeko_rl.experiments.coin_neutral_start import neutral_env
        # geom=None ⇒ the canonical E0 CONCAVE_CLAMP eval robot (unchanged). A robot VARIANT (BALLTIP matched-panel
        # regression) passes geom="POINT" + a ball-tip arm_mjcf_transform to swap the clamp for a spherical tip.
        self.env, self.cf = neutral_env(prefix_steps=0, geom=geom, arm_mjcf_transform=arm_mjcf_transform)
        self.inner = self.cf._env
        self.inner.reward_spec = RewardSpec.from_hymeko(CANONICAL_REWARD_FILE)      # v3 reward authority
        self.inner.reward_file = CANONICAL_REWARD_FILE
        self.horizon = int(horizon)
        self._t = 0
        self._strict = 0
        self._touched = False

    def _dtz(self) -> float:
        return float(self.inner._planar_metrics.disk_to_zone)

    def _speed(self) -> float:
        return float(np.linalg.norm(self.inner.data.qvel[self.inner._disk_x_adr:self.inner._disk_x_adr + 2]))

    def obs(self) -> np.ndarray:
        return np.asarray(self.inner.node_features(), np.float32).flatten()

    def reset(self, seed: int) -> np.ndarray:
        self.env.set_stage(0)
        self.env.reset(seed=int(seed))
        self._t = 0
        self._strict = 0
        self._touched = False
        return self.obs()

    def step(self, a4: np.ndarray):
        a = np.clip(np.asarray(a4, np.float32), -4.0, 4.0)
        self.inner.step(a)
        self._t += 1
        dtz = self._dtz()
        m = self.inner._planar_metrics
        self._touched = self._touched or bool(m.left_contact or m.right_contact)
        strict_ok = (dtz <= CENTER_TOL) and (self._speed() < SETTLE_VEL)
        self._strict = self._strict + 1 if strict_ok else 0
        self.inner._success = int(self._strict)                       # canonical v3 terminal fires at K
        self.inner.success_steps = HELD_DWELL
        reward = float(self.inner.reward_spec.evaluate(self.inner, dtz, self.inner.data.ctrl))   # canonical v3 reward
        delivered = bool(self._strict >= HELD_DWELL and self._touched)
        terminated = delivered
        truncated = (not terminated) and self._t >= self.horizon
        return self.obs(), reward, terminated, truncated, {"dtz": dtz, "strict": self._strict, "delivered": delivered}
