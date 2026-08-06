"""Embodiment-agnostic Vukobratović-ZMP stability core — the shared CIP-0 capturability certificate.

The proper Zero-Moment Point (Vukobratović 1969), including the angular-momentum-rate term ``Ḣ`` that
fires for rotational tips, plus the support-polygon criterion, are **embodiment-agnostic**: whole-body
CoM + momentum from MuJoCo ``subtree_*`` and a support region from the ground-contact bodies. The **same**
core certifies the AIBO's turning (paws, stance-weighted support) and the humanoid's balance (feet,
support box). Lifting it here makes it one control primitive both scenarios import — the CIP-0 stability
layer across embodiments.
"""

from __future__ import annotations

import numpy as np

from hymeko_control.cip.certificate import Certificate
from hymeko_control.language.schema_v0 import CertificateKind


def vukobratovic_zmp(model, data, prev_linvel: np.ndarray, prev_angmom: np.ndarray,
                     dt: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The Vukobratović ZMP ``(x, y)`` on flat ground, with the angular-momentum-rate term ``Ḣ``.

    ``ZMP = CoM_xy − (m·z·a_xy + [Ḣ_y, −Ḣ_x]) / (m·(z̈ + g))``. Whole-body CoM/momentum from MuJoCo
    ``subtree_com[0]/subtree_linvel[0]/subtree_angmom[0]``; ``a`` and ``Ḣ`` by finite difference against the
    previous step. # Preconditions ``dt > 0``; the model has the ``subtree_*`` fields. # Postconditions
    returns ``(zmp_xy, linvel, angmom)`` — thread the latter two as ``prev_*`` on the next step."""
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


def support_margin_box(data, zmp_xy: np.ndarray, bodies: "list[int]",
                       half_extent: "tuple[float, float]") -> float:
    """Signed distance from the ZMP to the axis-aligned support box (contact bodies' bbox + ``half_extent``).

    ``> 0`` ⇔ ZMP inside support ⇔ balanced. For feet with a finite sole (bipedal ZMP criterion)."""
    fp = np.array([data.xpos[b][:2] for b in bodies])
    lo = fp.min(0) - np.array(half_extent)
    hi = fp.max(0) + np.array(half_extent)
    return float(min(zmp_xy[0] - lo[0], hi[0] - zmp_xy[0], zmp_xy[1] - lo[1], hi[1] - zmp_xy[1]))


def support_margin_weighted(data, zmp_xy: np.ndarray, bodies: "list[int]", *,
                            contact_height: float = 0.06) -> float:
    """Signed distance from the ZMP to a stance-weighted support region — support ∝ how planted each
    contact body is (``max(0, contact_height − z)``). For point-foot contacts (paws) with no sole area.
    ``> 0`` ⇔ ZMP inside support. # Postconditions ``-1`` if all contacts are airborne (no support)."""
    feet = np.array([data.xpos[b][:2] for b in bodies])
    w = np.maximum(0.0, contact_height - np.array([data.xpos[b][2] for b in bodies]))
    if w.sum() < 1e-6:
        return -1.0
    c = (feet * w[:, None]).sum(0) / w.sum()
    rad = float(np.sqrt(((feet - c) ** 2).sum(1) * w).sum() / w.sum())
    return float(rad - float(np.hypot(*(zmp_xy - c))))


def capture_point(model, data) -> np.ndarray:
    """The LIPM capture point ``ξ = CoM_xy + CoM_vel_xy·√(CoM_z / g)`` — where the CoM would come to rest.

    The translational capturability measure (Pratt/Koolen): if ``ξ`` is inside the support polygon the robot
    can stop WITHOUT stepping (0-step capturable); if outside, it must step. # Postconditions ``(2,)``."""
    g = float(-model.opt.gravity[2]) or 9.81
    com = np.asarray(data.subtree_com[0])
    vel = np.asarray(data.subtree_linvel[0])
    return com[:2] + vel[:2] * float(np.sqrt(max(com[2], 1e-3) / g))


def capturability_level(data, cp: np.ndarray, bodies: "list[int]",
                        foot_half: "tuple[float, float]", max_step: float) -> "tuple[int, float, float]":
    """Pratt/Koolen N-step capturability of the capture point ``cp``:
    ``0`` = 0-step capturable (``cp`` in the support box: balance in place),
    ``1`` = 1-step capturable (``cp`` in support ⊕ one step of reach ``max_step``: MUST step),
    ``2`` = not capturable within one step (fall). # Postconditions returns ``(level, m0, m1)`` — the
    signed margins to the 0-step and 1-step boundaries (``> 0`` = inside)."""
    m0 = support_margin_box(data, cp, bodies, foot_half)
    m1 = support_margin_box(data, cp, bodies, (foot_half[0] + max_step, foot_half[1] + max_step))
    level = 0 if m0 > 0.0 else (1 if m1 > 0.0 else 2)
    return level, m0, m1


def zmp_support_certificate(name: str = "zmp_in_support") -> Certificate:
    """CIP-0 SAFETY certificate: the Vukobratović ZMP stays inside the support polygon throughout.

    Passes iff every ``zmp_margin`` in the trace's signals is ``> 0`` (genuine capturability, not mere
    survival). Reward-independent — the embodiment-agnostic stability certificate."""

    def _fn(_state, trace) -> bool:
        margins = [float(s.get("zmp_margin", -1.0)) for s in getattr(trace, "signals", [])]
        return bool(margins) and min(margins) > 0.0

    return Certificate(name, CertificateKind.SAFETY, _fn)
