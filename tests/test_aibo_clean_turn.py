"""The swing-lifted in-place turn — lifting the rotational-couple's swing feet gives a cleaner approach.

The rotational-couple turn drags its swing feet, and on wide goals the body spirals so it enters the goal
facing ~180° ("it goes in from behind", user, 2026-07-30). Lifting the swing feet at the right phase
(turn_swing_lift/turn_lift_off/turn_freq) makes the turn UPRIGHT without the crouch+widen stabilization and
lets the AIBO turn to FACE the goal then walk straight in — measured mean |heading| at reach ~12° vs ~24°
for the drifting-turn scaffold (a controlled comparison; the swing-lift, not just the tighter align, is
load-bearing: with the same align+no-stab the drifting turn reaches only 1/7 vs the lift turn's 6/7).
Honest scope: this is an END-TO-END arrival improvement, not a big single-turn drift cut (the max lateral
excursion of a bare turn is similar ~12 cm); the benefit is upright-without-stab + a cleaner homing walk.
Defaults reproduce the prior drifting turn exactly.
"""

from __future__ import annotations

import numpy as np

from scenarios.aibo.locomotion_gait import RotationalTurnGait, body_yaw, heading_error
from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


def _turn_upright(gait: RotationalTurnGait, turn: float = 1.0, steps: int = 1000) -> "tuple[float, bool]":
    """Return (total yaw deg, upright) for a pure turn from rest — the swing-lift stays upright on its own."""
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat"), seed=0)
    e = env._env
    e.reset(seed=1)
    yaw0 = body_yaw(e)
    up = True
    for _ in range(steps):
        e.step(gait.action(e, turn=turn))
        if e.data.xmat[e.torso].reshape(3, 3)[2, 2] < 0.5:
            up = False
            break
    return float(np.degrees(abs(body_yaw(e) - yaw0))), up


def _approach_heading(**cfg_fields) -> "tuple[float, int]":
    """Mean |heading| (deg) at aligned-reach and count reached, for the a=0 scaffold over a wide-bearing grid."""
    cfg = ResidualTrotConfig(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk",
                             require_facing_deg=25, max_steps=1600, **cfg_fields)
    env = ResidualTrotEnv(cfg, seed=0)
    e = env._env
    hs = []
    for bearing in (0, 40, -40, 90, -90, 135, -135):
        e.reset(seed=505)
        tx, ty = float(e.data.xpos[e.torso, 0]), float(e.data.xpos[e.torso, 1])
        yaw = body_yaw(e)
        b = np.deg2rad(bearing)
        e.goal = np.array([tx + 0.6 * np.cos(yaw + b), ty + 0.6 * np.sin(yaw + b)], np.float32)
        env._prev_dist = float(e.dist_to_goal())
        for _ in range(2400):
            env._apply(np.zeros(4, np.float32))
            if float(e.dist_to_goal()) <= 0.12 and abs(np.degrees(heading_error(e))) <= 25:
                hs.append(abs(float(np.degrees(heading_error(e)))))
                break
    return (float(np.mean(hs)) if hs else 90.0), len(hs)


def test_swing_lifted_turn_is_upright_without_stabilization() -> None:
    """The swing-lifted turn stays upright on its own — no crouch+widen needed, unlike the bare fast turn."""
    yaw, up = _turn_upright(RotationalTurnGait(swing_lift=0.35, lift_off=2.9, freq=1.6))
    assert up and yaw > 15.0                                  # a real turn, and it did not tip


def test_swing_lift_turn_beats_drifting_turn_end_to_end() -> None:
    """Controlled: the swing-lift turn arrives more head-on AND reaches more than the drifting turn (same align).

    The swing-lift — not just a tighter align — is load-bearing: with the same align+no-stab the DRIFTING
    turn barely reaches, while the LIFT turn reaches most bearings, and more head-on than the old drift+stab
    scaffold. Pins the 'goes in from behind' fix (mean |heading| ~24° -> ~12°)."""
    old_mean, old_n = _approach_heading(turn_rate=1.3, turn_align_deg=20, stab_crouch=0.5, stab_widen=0.4)
    lift_mean, lift_n = _approach_heading(turn_rate=1.2, turn_align_deg=15,
                                          turn_swing_lift=0.35, turn_lift_off=2.9, turn_freq=1.6)
    drift_mean, drift_n = _approach_heading(turn_rate=1.2, turn_align_deg=15)   # drifting turn, same align/no-stab
    assert lift_n >= 5 and lift_mean < 18.0                   # reaches most, head-on (measured 6/7 @ ~12°)
    assert lift_mean < old_mean                               # cleaner than the old drift+stab scaffold (~24°)
    assert lift_n > drift_n + 2                               # the swing-lift (not the align) does the work


def test_align_lever_repurposes_crouch_slot_gated() -> None:
    """align_res_scale>0 turns the stab residual's dead crouch slot (r[1]) into the turn-vs-walk ALIGN lever.

    On the swing-lift scaffold crouch+widen are dead DOF; the live lever that faces wide bearings is the
    align threshold (per-bearing-align oracle 0.75→0.93 held-out). This checks the lever is wired + gated:
    a=0 is the scaffold, r[1] changes the align (hence the rollout), and default (0) leaves crouch behaviour.
    """
    common = dict(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk", turn_rate=1.2,
                  turn_swing_lift=0.35, turn_lift_off=2.9, turn_freq=1.6, turn_align_deg=15,
                  align_res_scale=8.0, require_facing_deg=25, max_steps=1600)
    env = ResidualTrotEnv(ResidualTrotConfig(**common), seed=0)
    md0 = env.rollout_min_dist(lambda o: np.zeros(4, np.float32), (0.6, 90), seed=707, horizon=2400)[0]
    md1 = env.rollout_min_dist(lambda o: np.array([0.0, -1.0, 0.0, 0.0], np.float32),                # r[1] → align 7°
                               (0.6, 90), seed=707, horizon=2400)[0]
    assert md0 != md1                                                       # the align lever changes the outcome
    assert ResidualTrotConfig().align_res_scale == 0.0                      # default off → r[1] = crouch, unchanged


def test_turn_fields_default_to_prior_drifting_turn() -> None:
    """Regression: the low-drift turn is opt-in — defaults rebuild the prior rotational couple unchanged."""
    cfg = ResidualTrotConfig()
    assert cfg.turn_swing_lift == 0.0
    assert cfg.turn_freq == 0.0                              # 0 → falls back to gait_freq
    env = ResidualTrotEnv(cfg, seed=0)
    assert env._turn_gait.swing_lift == 0.0                  # drags its feet, as before
