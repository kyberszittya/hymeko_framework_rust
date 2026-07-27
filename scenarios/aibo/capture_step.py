"""Model-based capture-point protective step for the AIBO (lateral push recovery).

Unlike the sagittal humanoid (where a step underperforms the frontal PD — see
hymeko_humanoid/reports/2026-07-27-humanoid-lateral-step.md), the 22-DOF quadruped's
legs abduct freely, so a **capture-point step** is genuinely effective: when a lateral
push drives the LIPM capture point ``xi_y = com_y + com_y_vel·sqrt(com_z/g)`` toward the
edge of support, the controller widens the stance toward ``xi_y`` (abduct the legs apart,
scaled by the capture-point excursion) to place support under the falling COM.

Verified: this recovers the AIBO to rest (push-recovery Lyapunov V -> 0, the unchanged
generic certificate) for lateral pushes up to ~1.3 m/s, where the passive stand FALLS.
This is a scaffold (no RL); residual RL over it is future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hymeko_control.cip.certificate import Certificate
from hymeko_control.language.schema_v0 import CertificateKind

from .locomotion_gait import SteeredTrotGait
from .lyapunov import evaluate_lyapunov

_G = 9.81


def capture_point_y(env) -> float:
    """LIPM lateral capture point ``xi_y = com_y + com_y_vel·sqrt(com_z/g)``."""
    com = env.data.subtree_com[1]
    com_y_vel = float(env.data.qvel[1])                        # base lateral velocity ~ COM lateral velocity
    return float(com[1]) + com_y_vel * float(np.sqrt(max(float(com[2]), 0.05) / _G))


@dataclass(frozen=True)
class PushRecoveryLyapunov:
    """V for lateral push recovery: V -> 0 iff upright, at rest, COM back over the spawn axis."""

    w_up: float = 1.0
    w_v: float = 0.4
    w_off: float = 1.5

    def __call__(self, env) -> float:
        up = float(env._torso_uprightness())
        speed = float(np.hypot(*np.asarray(env.data.qvel)[:2]))
        com_lat = abs(float(env.data.subtree_com[1][1]))
        return 0.5 * (self.w_up * (1.0 - up) ** 2 + self.w_v * speed * speed + self.w_off * com_lat * com_lat)


@dataclass
class CapturePointStepper:
    """Reactive capture-point widening: abduct the legs toward ``xi_y`` to catch a lateral fall.

    # Preconditions env exposes ``_torso_uprightness``, ``ctrl_range``, the trot's PD fields,
    and 4 legs x {hip_abduct, hip_flex, knee}. # Postconditions ``action(env)`` returns a
    normalised ``(nu,)`` action in [-1, 1]: a PD stand-hold with a hip-abduction widening
    proportional to the capture-point excursion (0 when balanced -> auto-triggers only on a push).
    """

    gain: float = 4.0
    _gait: SteeredTrotGait = field(default_factory=SteeredTrotGait, init=False, repr=False)
    _abd: dict = field(default_factory=dict, init=False, repr=False)

    def _index(self, env) -> None:
        import mujoco
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.model.actuator_trnid[i, 0])
                 for i in range(env.model.nu)]
        self._abd = {n: i for i, n in enumerate(names) if "hip_abduct" in n}

    def action(self, env) -> np.ndarray:
        if not self._abd:
            self._index(env)
        a = self._gait.action(env, yaw_cmd=0.0, drive=0.0)            # PD stand-hold baseline
        com_y = float(env.data.subtree_com[1][1])
        g = float(np.clip(abs(capture_point_y(env) - com_y) * self.gain, 0.0, 1.0))
        for n, i in self._abd.items():                               # widen the stance apart (toward the fall)
            a[i] = g if n.endswith(("_fl", "_bl")) else -g
        return np.clip(a, -1.0, 1.0).astype(np.float32)


def push_recovery_certificate(name: str, V: PushRecoveryLyapunov, **kw) -> Certificate:
    """Generic reward-independent CIP-0 SAFETY certificate over a push-recovery V trace."""

    def _fn(_state, trace) -> bool:
        return evaluate_lyapunov([V(s) for s in trace.signals], **kw)["passes"]

    return Certificate(name, CertificateKind.SAFETY, _fn)


def recover_v_series(env, controller, push_vy: float, *, steps: int = 400,
                     V: PushRecoveryLyapunov | None = None) -> Sequence[float]:
    """Roll a lateral-push recovery under ``controller`` (a callable env->action) or None (stand).

    Returns the push-recovery V series. Applies ``push_vy`` (base lateral velocity) at reset.
    """
    import mujoco
    V = V or PushRecoveryLyapunov()
    env.reset(seed=0)
    env.data.qvel[1] = push_vy
    mujoco.mj_forward(env.model, env.data)
    stand = SteeredTrotGait()
    vs = []
    for _ in range(steps):
        a = controller.action(env) if controller is not None else stand.action(env, yaw_cmd=0.0, drive=0.0)
        _o, _r, term, trunc, _i = env.step(a)
        vs.append(V(env))
        if term or trunc:
            break
    return vs
