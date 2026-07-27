"""Realistic-motion contract for the AIBO — a directional joint-velocity governor.

The capture-point "recovery" exploited unphysical dynamics (leg joints at 26.9 rad/s, the
coin's ~27 rad/s failure mode). Real Aibo ERS-1000 servos do a few rad/s, not tens. This
governor caps the *accelerating* torque when a joint is at/over the velocity limit, while
**preserving braking** (so the robot can still stop) — the same principle as the coin's
``REALISTIC_MOTION_CONTRACT_V1`` (rebuilt here; the coin module lives in the main tree,
off-limits). Wrap any controller's normalised action with ``govern`` before stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointVelocityGovernor:
    """Directional velocity governor: zero the accelerating action of any joint at |v| >= v_max.

    # Preconditions env exposes ``model.actuator_trnid``/``model.jnt_dofadr`` and ``data.qvel``;
    ``action`` is the normalised per-actuator command in [-1, 1] (maps to +/-ctrl_range torque).
    # Postconditions returns a governed action: braking components are untouched, accelerating
    components on over-speed joints are zeroed -> joint speeds stay near ``v_max`` (realistic).
    """

    v_max: float = 8.0                       # rad/s — realistic servo ceiling (was exploited at 26.9)
    _dofs: list = field(default_factory=list, init=False, repr=False)

    def _index(self, env) -> None:
        self._dofs = [int(env.model.jnt_dofadr[env.model.actuator_trnid[i, 0]])
                      for i in range(env.model.nu)]

    def govern(self, env, action) -> np.ndarray:
        if not self._dofs:
            self._index(env)
        a = np.asarray(action, np.float64).copy()
        qv = np.asarray(env.data.qvel)[self._dofs]
        a[(qv >= self.v_max) & (a > 0.0)] = 0.0      # would accelerate past +v_max -> cut (keep braking)
        a[(qv <= -self.v_max) & (a < 0.0)] = 0.0     # would accelerate past -v_max -> cut
        return a.astype(np.float32)

    def max_joint_speed(self, env) -> float:
        if not self._dofs:
            self._index(env)
        return float(np.max(np.abs(np.asarray(env.data.qvel)[self._dofs])))
