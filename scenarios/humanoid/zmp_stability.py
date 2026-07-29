"""Humanoid balance certificate — thin embodiment adapter over the shared Vukobratović-ZMP core.

The proper Vukobratović ZMP (with ``Ḣ``) and the support-polygon certificate live once in
:mod:`hymeko_control.stability` — the same core the AIBO turning certificate uses. Here we bind it to the
humanoid: the feet (``foot_l/foot_r``) as the support with a foot sole extent (bipedal ZMP criterion).
This makes the previously *vacuous* `support_margin` certificate genuine, and distinguishes stability
(ZMP in support) from mere survival (not yet fallen) — the humanoid's own `SURVIVES_PARTIALLY_NOT_STABLE`.
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from hymeko_control.stability import support_margin_box, vukobratovic_zmp, zmp_support_certificate

_FOOT_NAMES = ("foot_l", "foot_r")
_FOOT_HALF = (0.09, 0.05)                                     # foot half-extent (x forward, y lateral)

# re-export the shared core so callers can `from scenarios.humanoid.zmp_stability import vukobratovic_zmp`
__all__ = ["vukobratovic_zmp", "foot_bodies", "foot_support_margin", "zmp_balance_certificate",
           "zmp_margin_series", "certified_zmp_margin"]

zmp_balance_certificate = zmp_support_certificate                # the humanoid's genuine support certificate


def foot_bodies(model: object) -> list[int]:
    return [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)) for nm in _FOOT_NAMES]


def foot_support_margin(data, zmp_xy: np.ndarray, feet: "list[int]",
                        foot_half: "tuple[float, float]" = _FOOT_HALF) -> float:
    """ZMP-to-foot-support margin (feet bbox + sole half-extent); ``> 0`` ⇔ ZMP in support ⇔ balanced."""
    return support_margin_box(data, zmp_xy, feet, foot_half)


def zmp_margin_series(env, controller, *, steps: int = 400, seed: int = 0,
                      action_dim: int | None = None) -> "list[float]":
    """Roll a balance episode under ``controller`` (env→action, or None = the PD scaffold a=0) and return
    the Vukobratović-ZMP support-margin series. # Postconditions each entry ``> 0`` iff the ZMP is in support."""
    feet = foot_bodies(env.model)
    env.reset(seed=seed)
    dt = float(env.model.opt.timestep) * int(getattr(env, "_frame_skip", getattr(env, "frame_skip", 10)))
    prev_v = np.asarray(env.data.subtree_linvel[0]).copy()
    prev_h = np.asarray(env.data.subtree_angmom[0]).copy()
    nu = action_dim if action_dim is not None else int(env.model.nu)
    margins: list[float] = []
    for _ in range(steps):
        a = controller(env) if controller is not None else np.zeros(nu, np.float32)
        _o, _r, term, trunc, _i = env.step(a)
        zmp, prev_v, prev_h = vukobratovic_zmp(env.model, env.data, prev_v, prev_h, dt)
        margins.append(foot_support_margin(env.data, zmp, feet))
        if term or trunc:
            break
    return margins


def certified_zmp_margin(env, *, perturb: float, steps: int = 400, seed: int = 0) -> "tuple[bool, float, Sequence[float]]":
    """Roll the PD scaffold (a=0) at a pitch perturbation; report (ZMP-in-support certifies, min margin, series)."""
    margins = zmp_margin_series(env, None, steps=steps, seed=seed)
    mn = min(margins) if margins else -1.0
    return mn > 0.0, mn, margins
