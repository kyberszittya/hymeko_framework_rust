"""Tests for verify_arm — generate-and-verify proper robot arms.

Positive: HyMeKo-emitted arms (galambos planar, reach arm) verify as proper (connected, articulate,
reachable). Negative: a hand-built arm whose link floats far from its joint is flagged disconnected.
The emitted cases need the CLI; the negative case does not.
"""
from __future__ import annotations

import pytest

from hymeko_rl.env.arm_world import _hymeko_cli, emit_arm_mjcf
from hymeko_rl.env.verify_arm import verify_arm


def _cli() -> bool:
    try:
        _hymeko_cli()
        return True
    except FileNotFoundError:
        return False


requires_cli = pytest.mark.skipif(not _cli(), reason="hymeko CLI not built")


def test_disconnected_arm_is_flagged() -> None:
    mjcf = """<mujoco><worldbody>
      <body name="base"><geom type="box" size="0.02 0.02 0.02"/>
        <body name="link" pos="0.5 0 0">
          <joint name="j" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.02 0.02 0.02"/>
        </body></body>
    </worldbody></mujoco>"""
    rep = verify_arm(mjcf, ee_body="link")
    assert rep.loads
    assert not rep.connected          # the joint floats 0.5 m from the base geom
    assert not rep.ok
    assert rep.max_link_gap > 0.1


def test_missing_ee_body_not_ok() -> None:
    mjcf = """<mujoco><worldbody><body name="b"><geom type="sphere" size="0.05"/></body></worldbody></mujoco>"""
    rep = verify_arm(mjcf, ee_body="nope")
    assert rep.loads and not rep.ok


@requires_cli
def test_emitted_planar_arm_is_proper() -> None:
    mjcf = emit_arm_mjcf("data/robotics/galambos_planar.hymeko", name="g", control_mode="position")
    rep = verify_arm(mjcf, ee_body="lower_left")
    assert rep.ok, rep.summary()
    assert rep.connected and rep.articulates
    assert rep.reach_radius > 0.2     # the fingertip sweeps a real workspace


@requires_cli
def test_emitted_reach_arm_is_proper() -> None:
    mjcf = emit_arm_mjcf("data/robotics/reach_arm.hymeko", name="reach", control_mode="position")
    rep = verify_arm(mjcf, ee_body="flange")
    assert rep.ok, rep.summary()
    assert rep.connected and rep.articulates
