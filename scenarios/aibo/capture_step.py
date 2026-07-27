"""Model-based capture-point stance-WIDENING (sprawl reflex) for the AIBO lateral push recovery.

HONEST NAMING (corrected): this is **not a step**. Measured, the controller abducts ALL
four legs outward roughly symmetrically (front paws ~0.36 m, back ~0.25 m; >=1 foot off
209/300 steps) — a startle-like *sprawl* that lowers the COM and widens the base toward the
LIPM capture point ``xi_y = com_y + com_y_vel·sqrt(com_z/g)``, catching the lateral fall. A
real protective *step* (keep 3 legs planted, lift + swing + place ONE leg) does NOT work
here: commanding a single-leg swing makes all four feet leave the ground (max 4 off) and
fails to certify (Vfinal ~0.72). So the quadruped recovers by widening, not stepping.

DYNAMICS EXPLOIT (retracted): the widening "recovery" is ALSO physically unrealistic — the
leg joints hit 26.9 rad/s (real Aibo ~3-8; ~ the coin's 27.2 rad/s exploit), the base
launches at 0.98 m/s, and all four feet are airborne 113/300 steps. It certifies only
because the model has no slew/torque/contact governor (REALISTIC_MOTION_CONTRACT_V1 was
never applied to this line). So the certificate pass is NOT robot-transferable. A valid
protective response needs (a) a realistic motion contract, then (b) contact-scheduled
dynamically-balanced stepping (whole-body MPC / a learned gait). Both unaddressed here.
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
class CapturePointWidening:
    """Reactive capture-point stance-WIDENING (sprawl reflex) — NOT a step.

    Abducts all legs apart toward ``xi_y`` to widen the base under a lateral fall. Measured to
    be a symmetric sprawl (all four paws splay), not a single-leg step; a real step fails here
    (see module docstring).

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
