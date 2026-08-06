"""AIBO turning stability — thin embodiment adapter over the shared Vukobratović-ZMP core.

The proper Vukobratović ZMP (with ``Ḣ``) and the support-polygon certificate live once in
:mod:`hymeko_control.stability` — the same core the humanoid balance certificate uses. Here we bind it to
the AIBO turn: the paws as point-foot contacts (stance-weighted support). A turn is CIP-0-SAFETY certified
iff the ZMP stays in support throughout — under the motion contract (governor). Validation: the stable
rotational-couple turn certifies (≤61°/1000), the fast turn's ZMP leaves support and it tips — the
~47–61°/1000 ceiling as a Vukobratović-ZMP / Lyapunov capturability boundary.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from hymeko_control.stability import support_margin_weighted, vukobratovic_zmp, zmp_support_certificate

_PAW_NAMES = ("paw_fl", "paw_fr", "paw_bl", "paw_br")
TurnFn = Callable[[object], np.ndarray]

# re-export the shared core (back-compat: callers import these from here)
__all__ = ["vukobratovic_zmp", "support_margin", "turn_zmp_margins", "zmp_stability_certificate", "_paw_bodies"]

zmp_stability_certificate = zmp_support_certificate             # the AIBO turn's genuine support certificate


def _paw_bodies(model: object) -> list[int]:
    return [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)) for nm in _PAW_NAMES]


def support_margin(env, zmp_xy: np.ndarray, paws: "list[int]") -> float:
    """ZMP-to-support margin for the AIBO's point-foot paws (stance-weighted); ``> 0`` ⇔ ZMP in support.

    ``env`` may be the MuJoCo data or an object exposing ``.data`` — accepts both for back-compat."""
    data = getattr(env, "data", env)
    return support_margin_weighted(data, zmp_xy, paws)


def turn_zmp_margins(env, turn_fn: TurnFn, *, steps: int = 400, seed: int = 0) -> "list[float]":
    """Roll a turn under ``turn_fn`` (env→action) and return the Vukobratović-ZMP support-margin series.

    # Preconditions ``turn_fn`` returns a normalized action; env exposes MuJoCo model/data + frame_skip.
    # Postconditions ``len == steps`` (or fewer if it terminates); each entry ``> 0`` iff ZMP in support."""
    paws = _paw_bodies(env.model)
    env.reset(seed=seed)
    dt = float(env.model.opt.timestep) * int(env.frame_skip)
    prev_v = np.asarray(env.data.subtree_linvel[0]).copy()
    prev_h = np.asarray(env.data.subtree_angmom[0]).copy()
    margins: list[float] = []
    for _ in range(steps):
        env.step(turn_fn(env))
        zmp, prev_v, prev_h = vukobratovic_zmp(env.model, env.data, prev_v, prev_h, dt)
        margins.append(support_margin_weighted(env.data, zmp, paws))
    return margins
