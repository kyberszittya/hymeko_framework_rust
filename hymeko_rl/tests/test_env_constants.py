"""Tests for the ontological env constants (``hymeko_rl/env/constants.py``) — Phase 1.

Three layers, matching the plan (docs/plans/2026-07-04-env-constants-ontology/):
- **values-parity**: each named constant / string helper equals the literal it replaced (guards a
  typo in the extraction);
- **collision-scheme**: the MuJoCo bitmask predicate over the named channels behaves as designed —
  a regression that fails against a wrong mask (the coin must stay unreachable by arm links);
- **golden guard**: the emitted MJCF still carries byte-identical option/collision fragments and
  compiles to a valid ``MjModel`` (behaviour-preserving).
"""
from __future__ import annotations

import mujoco
import pytest

from hymeko_rl.env.constants import Collision, Physics
from hymeko_rl.env.planar_grasp_env import compose_planar_scene, make_planar_arms_mjcf


# ── values-parity ────────────────────────────────────────────────────────────
def test_physics_values() -> None:
    assert Physics.TIMESTEP == pytest.approx(2e-3)
    assert Physics.STABLE_DT == pytest.approx(5e-4)
    assert Physics.GRAVITY == (0.0, 0.0, -9.81)
    assert Physics.INTEGRATOR == "implicitfast"


def test_physics_attr_strings_match_old_literals() -> None:
    assert Physics.gravity_attr() == "0 0 -9.81"
    assert Physics.option_attrs() == 'timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"'


def test_collision_attr_strings_match_old_literals() -> None:
    assert Collision.attr(Collision.FINGERTIP) == 'contype="1" conaffinity="3"'
    assert Collision.attr(Collision.COIN) == 'contype="2" conaffinity="2"'
    assert Collision.attr(Collision.VISUAL) == 'contype="0" conaffinity="0"'
    assert int(Collision.Affinity.ANY) == 3   # the floor's bare conaffinity="3"


# ── collision-scheme (the design invariant) ──────────────────────────────────
def test_collision_scheme_isolates_coin_from_arm_links() -> None:
    C = Collision
    assert C.collide(C.COIN, C.FINGERTIP) is True      # a fingertip moves the coin
    assert C.collide(C.COIN, C.FLOOR) is True           # the coin rests on the floor
    assert C.collide(C.COIN, C.ARM_DEFAULT) is False    # arm links CANNOT touch the coin — the whole point
    assert C.collide(C.FINGERTIP, C.FLOOR) is True       # fingertips touch the floor
    # finger-on-finger is NOT masked out (it is detected separately as a crash, not prevented):
    assert C.collide(C.FINGERTIP, C.FINGERTIP) is True
    # regression: a mask typo giving the coin the arm's affinity would let an arm knock it — must not.
    assert C.collide(C.COIN, (int(C.Type.DEFAULT), int(C.Affinity.DEFAULT))) is False


# ── golden guard: emitted MJCF unchanged + compiles ──────────────────────────
def test_planar_mjcf_carries_expected_fragments_and_compiles() -> None:
    arms = make_planar_arms_mjcf()
    assert '<option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>' in arms
    assert 'contype="1" conaffinity="3"' in arms          # fingertip channel unchanged
    assert 'conaffinity="3"/>' in arms                     # floor's bare conaffinity unchanged
    scene = compose_planar_scene(arms)
    assert 'contype="2" conaffinity="2"' in scene          # coin channel unchanged
    model = mujoco.MjModel.from_xml_string(scene)          # build regression: valid MjModel
    assert model.opt.timestep == pytest.approx(2e-3)
