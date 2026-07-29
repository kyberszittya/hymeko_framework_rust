"""Vukobratović-ZMP turning-stability certificate — the Lyapunov/capturability boundary for fast turning.

The rotational-couple turn tips above ~47°/1000 steps. This formalizes *why* with the proper
**Vukobratović Zero-Moment Point**: the point where the ground-reaction moment has no horizontal
component. Unlike the LIPM capture point (translational, in :mod:`capture_step`), the full ZMP includes
the **angular-momentum-rate** term ``Ḣ`` — the term that fires for a *rotational* (spin) tip, which the
capture point misses. The stability boundary is **ZMP ∈ support polygon** (Vukobratović's criterion); the
Lyapunov region of attraction (capturability) is the set of states from which the ZMP can be kept inside.

A turn is CIP-0-SAFETY certified iff the ZMP stays within the (stance-weighted) support throughout — a
formal, reward-independent stability boundary over the *governed* dynamics (unlike the retracted exploit
in :mod:`capture_step`, this runs under the motion contract). Validation: the stable turn certifies, the
fast turn does not — the ~47°/1000 ceiling as a Vukobratović-ZMP certificate.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from hymeko_control.cip.certificate import Certificate
from hymeko_control.language.schema_v0 import CertificateKind

_PAW_NAMES = ("paw_fl", "paw_fr", "paw_bl", "paw_br")
TurnFn = Callable[[object], np.ndarray]


def _paw_bodies(model: object) -> list[int]:
    return [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)) for nm in _PAW_NAMES]


def vukobratovic_zmp(env, prev_linvel: np.ndarray, prev_angmom: np.ndarray,
                     dt: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The Vukobratović ZMP ``(x, y)`` on flat ground, including the angular-momentum-rate term ``Ḣ``.

    ``ZMP = CoM_xy − (m·z·a_xy + [Ḣ_y, −Ḣ_x]) / (m·(z̈ + g))``. Whole-body CoM, momentum from MuJoCo
    ``subtree_*``; ``a`` and ``Ḣ`` by finite difference against the previous step. # Postconditions
    returns ``(zmp_xy, linvel, angmom)`` — the latter two to thread as ``prev_*`` next step."""
    m = float(np.asarray(env.model.body_mass).sum())
    g = float(-env.model.opt.gravity[2]) or 9.81
    com = np.asarray(env.data.subtree_com[0])
    v = np.asarray(env.data.subtree_linvel[0])
    h = np.asarray(env.data.subtree_angmom[0])
    a = (v - prev_linvel) / dt
    hdot = (h - prev_angmom) / dt
    fz = m * (a[2] + g)
    if abs(fz) < 1e-6:
        fz = m * g
    zmp_x = com[0] - (m * com[2] * a[0] + hdot[1]) / fz
    zmp_y = com[1] - (m * com[2] * a[1] - hdot[0]) / fz
    return np.array([zmp_x, zmp_y]), v.copy(), h.copy()


def support_margin(env, zmp_xy: np.ndarray, paws: "list[int]") -> float:
    """Signed distance from the ZMP to the stance-weighted support region: ``> 0`` = ZMP inside = stable.

    Support = the planted feet (paws below 6 cm) weighted by how planted; the margin is the support
    "radius" minus the ZMP's distance from the support centroid. # Postconditions ``> 0`` stable."""
    feet = np.array([env.data.xpos[b][:2] for b in paws])
    w = np.maximum(0.0, 0.06 - np.array([env.data.xpos[b][2] for b in paws]))
    if w.sum() < 1e-6:
        return -1.0                                          # no support (all airborne) → unstable
    c = (feet * w[:, None]).sum(0) / w.sum()
    rad = float(np.sqrt(((feet - c) ** 2).sum(1) * w).sum() / w.sum())
    return float(rad - float(np.hypot(*(zmp_xy - c))))


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
        zmp, prev_v, prev_h = vukobratovic_zmp(env, prev_v, prev_h, dt)
        margins.append(support_margin(env, zmp, paws))
    return margins


def zmp_stability_certificate(name: str = "turn_zmp_support") -> Certificate:
    """CIP-0 SAFETY certificate: the turn keeps the Vukobratović ZMP inside the support polygon.

    Passes iff every margin in the trace's ``zmp_margin`` signal is ``> 0`` (ZMP never leaves support)."""

    def _fn(_state, trace) -> bool:
        margins = [float(s.get("zmp_margin", -1.0)) for s in trace.signals]
        return bool(margins) and min(margins) > 0.0

    return Certificate(name, CertificateKind.SAFETY, _fn)
