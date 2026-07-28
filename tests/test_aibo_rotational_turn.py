"""RotationalTurnGait + turn_then_walk — the goal-reaching turning fix.

The scripted skid-steer (SteeredTrotGait) barely rotates (~19°/1000 steps). The rotational-couple turn
(diagonal legs stride in OPPOSITE directions) generates a real yaw couple — measured much stronger and
directional. These lock: the couple is directional (turn sign = yaw sign) and stronger than the skid-
steer; the scaffold's ``turn_then_walk`` heading mode uses it when the heading error is wide; and it
reaches a wide-bearing goal that the default ``arc`` mode cannot.
"""

from __future__ import annotations

import numpy as np

from scenarios.aibo.locomotion_gait import RotationalTurnGait, SteeredTrotGait, body_yaw
from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


def _yaw_rate(gait, *, steps: int = 500, **kw) -> float:
    """World-yaw change (deg per 1000 steps) commanding a constant turn/steer for ``steps``."""
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat"), seed=0)._env
    env.reset(seed=0)
    y0 = body_yaw(env)
    for _ in range(steps):
        env.step(gait.action(env, **kw))
    dyaw = float(np.arctan2(np.sin(body_yaw(env) - y0), np.cos(body_yaw(env) - y0)))
    return np.degrees(dyaw) * 1000.0 / steps


def test_rotational_turn_is_directional() -> None:
    ccw = _yaw_rate(RotationalTurnGait(), turn=1.0)
    cw = _yaw_rate(RotationalTurnGait(), turn=-1.0)
    assert ccw > 0 and cw < 0                              # turn sign sets the yaw direction
    assert abs(ccw + cw) < 0.5 * max(abs(ccw), abs(cw))    # roughly antisymmetric in the turn sign


def test_rotational_turn_beats_skid_steer() -> None:
    rot = abs(_yaw_rate(RotationalTurnGait(), turn=1.0))
    skid = abs(_yaw_rate(SteeredTrotGait(), yaw_cmd=0.6, drive=1.0))
    assert rot > 2.0 * skid                                # the couple is a real turn, not a weak arc


def test_turn_then_walk_defaults_off() -> None:
    assert ResidualTrotConfig().heading_mode == "arc"      # default = prior behaviour


def test_base_gait_uses_turn_when_heading_wide() -> None:
    arc = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", heading_mode="arc"), seed=0)
    ttw = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", heading_mode="turn_then_walk"), seed=0)
    for e in (arc, ttw):
        e.reset(seed=707)
        r = e._env
        tx, ty = float(r.data.xpos[r.torso, 0]), float(r.data.xpos[r.torso, 1])
        yaw = float(np.arctan2(r.data.xmat[r.torso].reshape(3, 3)[1, 0], r.data.xmat[r.torso].reshape(3, 3)[0, 0]))
        r.goal = np.array([tx + 0.6 * np.cos(yaw + np.pi / 2), ty + 0.6 * np.sin(yaw + np.pi / 2)], np.float32)  # 90deg
        r._step = 100                                     # advance the gait phase off zero (at ph=0 both gaits coincide)
    herr = np.pi / 2
    a_arc = arc._base_gait_action(herr, pursuit=0.6, base_drive=1.0)
    a_ttw = ttw._base_gait_action(herr, pursuit=0.6, base_drive=1.0)
    assert not np.allclose(a_arc, a_ttw)                   # turn_then_walk takes the rotational-couple branch


def test_turn_then_walk_reaches_a_wide_goal_that_arc_misses() -> None:
    zfn = lambda o: np.zeros(4, np.float32)  # noqa: E731

    def reached(mode: str) -> bool:
        env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", heading_mode=mode), seed=0)
        md, ok, up = env.rollout_min_dist(zfn, (0.5, 40), seed=705, horizon=2400)
        return bool(ok and up > 0.5)

    assert reached("turn_then_walk")                       # the couple turns to face +40deg and walks in
    assert not reached("arc")                              # the skid-steer arcs past it
