"""turn_first scaffold knob — cut forward drive while the heading error is wide (turn-in-place intent).

Locks the config contract: off by default (backward-compatible walk-and-arc), and when on it caps the
forward drive once |heading error| exceeds the threshold. (Diagnostic finding: this alone does NOT fix
goal-reaching because the underlying skid-steer turning is too weak — see the turning report — but the
knob behaves as specified.)
"""

from __future__ import annotations

import numpy as np

from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


def _env(turn_first_deg: float) -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat",
                                              turn_first_deg=turn_first_deg, turn_drive=0.15), seed=0)


def test_turn_first_defaults_off() -> None:
    assert ResidualTrotConfig().turn_first_deg == 0.0                 # default = walk-and-arc, unchanged


def test_base_drive_zero_within_reach() -> None:
    env = _env(30.0)
    assert env._base_drive(np.deg2rad(90.0), dist=0.05) == 0.0        # inside reach radius -> stop


def test_turn_first_caps_drive_when_heading_wide() -> None:
    env = _env(30.0)
    assert env._base_drive(np.deg2rad(90.0), dist=0.6) == 0.15        # wide bearing -> capped to turn_drive


def test_turn_first_leaves_drive_when_heading_narrow() -> None:
    env = _env(30.0)
    assert env._base_drive(np.deg2rad(10.0), dist=0.6) == 1.0         # near-aligned -> full drive


def test_off_always_full_drive_outside_reach() -> None:
    env = _env(0.0)
    assert env._base_drive(np.deg2rad(90.0), dist=0.6) == 1.0         # turn_first off -> walk-and-arc
