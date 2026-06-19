"""Safety / configuration state of a reaching arm — the basis for the death conditions
(ground contact, self-collision) and the soft config penalties (joint-limit proximity,
below-ground).

Kept separate from ``arm_reach_env`` and ``reward`` so both import it without a cycle, and so
:func:`compute_safety` is unit-testable from a planted ``(model, data)`` with no env.

A non-floor contact counts as self-collision only between **non-adjacent** bodies — parent/child
and same-body pairs are excluded explicitly (the emitted scene does not reliably filter them via
``filterparent``, and adjacent links touch at their shared joint by construction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SafetyState:
    """Live safety/config state.

    * ``ground_contact`` — a robot geom touches the floor plane (a death condition).
    * ``self_collision`` — two non-adjacent robot geoms touch (a death condition).
    * ``joint_margin`` — min normalised distance to any joint limit, ``∈ [0, 1]``; ``0`` = at a
      limit, ``1`` = mid-range.
    * ``below_ground`` — the lowest moving-link body origin dipped below the floor plane.
    """

    ground_contact: bool
    self_collision: bool
    joint_margin: float
    below_ground: bool


# A clean state — no contact, mid-range, above ground. The reward terms fall back to this on an
# env that has not computed a safety state, so the terms are inert there (contribute 0).
CLEAN_SAFETY = SafetyState(ground_contact=False, self_collision=False,
                           joint_margin=1.0, below_ground=False)


def compute_safety(model: Any, data: Any, floor_geom_id: int, *,
                   below_tol: float = 1e-3) -> SafetyState:
    """Derive a :class:`SafetyState` from a stepped ``(model, data)``.

    # Preconditions ``data`` is post-``mj_step`` (contacts populated); ``floor_geom_id`` is the
      plane geom's id, or ``< 0`` if the scene has no floor.
    # Postconditions a finite :class:`SafetyState`; ``joint_margin ∈ [0, 1]``.
    """
    ground = False
    self_col = False
    parent = model.body_parentid
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if floor_geom_id >= 0 and (g1 == floor_geom_id or g2 == floor_geom_id):
            ground = True
            continue
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 == b2 or int(parent[b1]) == b2 or int(parent[b2]) == b1:
            continue  # same body or adjacent links touch at their joint — not a self-collision
        self_col = True

    margin = 1.0
    for j in range(int(model.njnt)):
        if not bool(model.jnt_limited[j]):
            continue
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        half = (hi - lo) / 2.0
        if half <= 0.0:
            continue
        q = float(data.qpos[int(model.jnt_qposadr[j])])
        margin = min(margin, max(0.0, min(q - lo, hi - q) / half))

    zmin = min(float(data.xpos[b, 2]) for b in range(1, int(model.nbody)))
    return SafetyState(ground_contact=ground, self_collision=self_col,
                       joint_margin=margin, below_ground=zmin < -below_tol)
