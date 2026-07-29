"""Vukobratović-ZMP balance certificate for the humanoid — the multi-embodiment capturability boundary.

The same proper Vukobratović ZMP (with the angular-momentum-rate term Ḣ) that certifies the AIBO's
turning stability (`hymeko_aibo` `scenarios/aibo/turn_stability.py`) also certifies the humanoid's
balance — the whole-body ZMP and the support-polygon criterion are **embodiment-agnostic**. This makes the
previously *vacuous* `support_margin` certificate (welded-base era) **genuine**: the ZMP must stay inside
the foot-support polygon.

It distinguishes genuine **stability** (ZMP in support) from mere **survival** (not yet fallen) — exactly
the humanoid's own `SURVIVES_PARTIALLY_NOT_STABLE` finding: a PD scaffold can stay upright while the ZMP
has already left support (no capturability margin). Validated monotone/predictive: PASS at pitch-rate ≤ 1,
FAIL at ≥ 2 (ZMP leaves support before the fall). Zero-moment point after Vukobratović (1969).
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from hymeko_control.cip.certificate import Certificate
from hymeko_control.language.schema_v0 import CertificateKind

_FOOT_NAMES = ("foot_l", "foot_r")
_FOOT_HALF = (0.09, 0.05)                                     # foot half-extent (x forward, y lateral) for the support box


def foot_bodies(model: object) -> list[int]:
    return [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)) for nm in _FOOT_NAMES]


def vukobratovic_zmp(model, data, prev_linvel: np.ndarray, prev_angmom: np.ndarray,
                     dt: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The Vukobratović ZMP ``(x, y)`` on flat ground, including the angular-momentum-rate term ``Ḣ``.

    ``ZMP = CoM_xy − (m·z·a_xy + [Ḣ_y, −Ḣ_x]) / (m·(z̈ + g))``. Whole-body CoM/momentum from MuJoCo
    ``subtree_*``; ``a`` and ``Ḣ`` by finite difference. Identical physics to the AIBO certificate — the
    multi-embodiment core. # Postconditions returns ``(zmp_xy, linvel, angmom)`` to thread as ``prev_*``."""
    m = float(np.asarray(model.body_mass).sum())
    g = float(-model.opt.gravity[2]) or 9.81
    com = np.asarray(data.subtree_com[0])
    v = np.asarray(data.subtree_linvel[0])
    h = np.asarray(data.subtree_angmom[0])
    a = (v - prev_linvel) / dt
    hdot = (h - prev_angmom) / dt
    fz = m * (a[2] + g)
    if abs(fz) < 1e-6:
        fz = m * g
    zmp_x = com[0] - (m * com[2] * a[0] + hdot[1]) / fz
    zmp_y = com[1] - (m * com[2] * a[1] - hdot[0]) / fz
    return np.array([zmp_x, zmp_y]), v.copy(), h.copy()


def foot_support_margin(data, zmp_xy: np.ndarray, feet: "list[int]",
                        foot_half: "tuple[float, float]" = _FOOT_HALF) -> float:
    """Signed distance from the ZMP to the foot-support box (feet bounding box + foot half-extent).

    ``> 0`` ⇔ ZMP inside the support polygon ⇔ balanced (Vukobratović's criterion). # Postconditions
    the margin decreases monotonically as the ZMP approaches / crosses the support boundary."""
    fp = np.array([data.xpos[b][:2] for b in feet])
    lo = fp.min(0) - np.array(foot_half)
    hi = fp.max(0) + np.array(foot_half)
    return float(min(zmp_xy[0] - lo[0], hi[0] - zmp_xy[0], zmp_xy[1] - lo[1], hi[1] - zmp_xy[1]))


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


def zmp_balance_certificate(name: str = "zmp_in_support") -> Certificate:
    """CIP-0 SAFETY certificate: the Vukobratović ZMP stays inside the foot-support polygon throughout.

    Passes iff every ``zmp_margin`` in the trace's signals is ``> 0`` (genuine capturability, not just
    survival). Reward-independent — the multi-embodiment stability certificate."""

    def _fn(_state, trace) -> bool:
        margins = [float(s.get("zmp_margin", -1.0)) for s in getattr(trace, "signals", [])]
        return bool(margins) and min(margins) > 0.0

    return Certificate(name, CertificateKind.SAFETY, _fn)


def certified_zmp_margin(env, *, perturb: float, steps: int = 400, seed: int = 0) -> "tuple[bool, float, Sequence[float]]":
    """Convenience: roll the PD scaffold (a=0) at a given pitch perturbation and report
    (ZMP-in-support certifies, min margin, series)."""
    margins = zmp_margin_series(env, None, steps=steps, seed=seed)
    mn = min(margins) if margins else -1.0
    return mn > 0.0, mn, margins
