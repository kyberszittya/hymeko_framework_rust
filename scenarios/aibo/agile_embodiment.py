"""Agile-embodiment semantics — widen the AIBO stance, and the propulsion/turning tradeoff it exposes.

The residual-RL and richer-primitive arcs both hit the AIBO turn/stability wall (stable turn ~15°/1000
steps; faster turning tips). That wall is a *model* property, so the lever is a **semantic (`.hymeko`)
change**, not more control. `widen_stance` moves the hip-abduction lateral offset outward (the stance
width) — a longer yaw moment arm + a wider base.

Measured result (`quadruped_agile.hymeko`, y = 0.062 → 0.11): turn authority **~4.6× (15 → 70°/1000)**
and **tipping eliminated** (stable through a full spin). BUT the same wide stance **cripples forward
propulsion** (the trot walks ~2.7× slower, and its net direction flips with stance width) — the AIBO
morphology has a **coupled propulsion/turning tradeoff** that a single geometric parameter does not
resolve. A truly agile AIBO needs co-designed morphology **and** a gait tuned/trained for it, not a
one-parameter edit. This module is the semantic transform + the diagnostic behind that finding.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .locomotion_gait import SteeredTrotGait

_BASE_HALF_WIDTH = 0.062  # the canonical AIBO hip-abduction lateral offset (data/robotics/quadruped.hymeko)


def widen_stance(hymeko_text: str, half_width: float) -> str:
    """Return ``hymeko_text`` with the four hip-abduction lateral offsets set to ``±half_width``.

    # Preconditions
    ``hymeko_text`` contains the four ``@hip_abduct_{leg}`` joints with a ``±0.062`` y-offset (the
    canonical quadruped). ``half_width > 0``. # Postconditions: each hip-abduction attaches at
    ``y = ±half_width`` (a wider/narrower stance); no other body is changed.
    """
    if half_width <= 0.0:
        raise ValueError(f"half_width must be > 0, got {half_width}")
    out = re.sub(r'(@hip_abduct_\w+:.*?\[\[\s*[-0-9.]+,\s*)0\.062', rf'\g<1>{half_width}', hymeko_text)
    out = re.sub(r'(@hip_abduct_\w+:.*?\[\[\s*[-0-9.]+,\s*)-0\.062', rf'\g<1>-{half_width}', out)
    return out


def write_agile_variant(source: Path, dest: Path, half_width: float = 0.11) -> Path:
    """Write a widened-stance variant of ``source`` to ``dest`` (same dir so the import resolves)."""
    dest.write_text(widen_stance(source.read_text(), half_width))
    return dest


def measure_forward_propulsion(env, gait: SteeredTrotGait, governor, steps: int = 700,
                               settle: int = 80, seed: int = 0) -> float:
    """Forward-walk displacement magnitude (m) of the trot over ``steps`` — the propulsion measure.

    Magnitude (not signed) because the trot's net direction is fragilely coupled to the geometry
    (it flips with stance width); we characterise how far it travels, either way.
    """
    env.reset(seed=seed)
    for _ in range(settle):
        env.step(governor.govern(env, gait.action(env, yaw_cmd=0.0, drive=0.0)))
    x0 = float(env.data.xpos[env.torso, 0])
    y0 = float(env.data.xpos[env.torso, 1])
    for _ in range(steps):
        env.step(governor.govern(env, gait.action(env, yaw_cmd=0.0, drive=1.0)))
    dx = float(env.data.xpos[env.torso, 0]) - x0
    dy = float(env.data.xpos[env.torso, 1]) - y0
    return float(np.hypot(dx, dy))
