"""Tests for render-time scene beautification (hymeko_rl/env/scene_style.py).

The contract: beautify_mjcf adds *visual* assets only — the output loads, carries skybox / floor /
lights / polished materials, and has the SAME bodies, joints, and actuators as the bare input.
Pure-string tests need no CLI/GL; the invariant test emits a real arm (CLI-gated, like the sibling
emit tests) and checks the hypergraph is untouched.
"""
from __future__ import annotations

import subprocess

import mujoco
import pytest

from hymeko_rl.env.arm_world import emit_arm_mjcf
from hymeko_rl.env.scene_style import SceneStyle, beautify_mjcf
from hymeko_rl.agents.hypergraph_state import HypergraphState

_MINIMAL = """<mujoco model="t">
  <asset><material name="mat_a" rgba="0.2 0.3 0.4 1"/></asset>
  <worldbody>
    <body name="b"><joint name="j" type="hinge"/>
      <geom type="box" size="0.1 0.1 0.1" material="mat_a"/></body>
  </worldbody>
</mujoco>"""

_NO_ASSET = """<mujoco model="t">
  <worldbody><body name="b"><geom type="box" size="0.1 0.1 0.1"/></body></worldbody>
</mujoco>"""


def _emit_arm_or_skip(robot: str) -> str:
    """Emit an arm MJCF, or skip if the hymeko CLI is not built (matches the sibling tests)."""
    try:
        return emit_arm_mjcf(robot, name="arm", control_mode="position")
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as err:
        pytest.skip(f"hymeko CLI unavailable: {err}")


def test_beautified_minimal_loads_and_has_assets() -> None:
    out = beautify_mjcf(_MINIMAL)
    mujoco.MjModel.from_xml_string(out)            # loads (well-formed)
    for token in ('type="skybox"', 'name="hk_grid_mat"', 'name="hk_floor"',
                  'name="hk_light0"', "<visual>"):
        assert token in out, f"missing {token}"


def test_polish_adds_specular_to_link_material() -> None:
    out = beautify_mjcf(_MINIMAL, SceneStyle(mat_specular=0.6))
    # the bare link material gains the polish; the floor material is unaffected by the link regex.
    assert 'name="mat_a"' in out and "specular=" in out.split('name="mat_a"')[1].split("/>")[0]


def test_creates_asset_block_when_absent() -> None:
    out = beautify_mjcf(_NO_ASSET)
    assert "<asset>" in out and 'type="skybox"' in out
    mujoco.MjModel.from_xml_string(out)


def test_no_worldbody_raises() -> None:
    with pytest.raises(ValueError, match="worldbody"):
        beautify_mjcf("<mujoco model='x'></mujoco>")


def test_beautify_preserves_kinematics_on_emitted_arm() -> None:
    """The no-physics-change invariant: same bodies/joints/actuators, same hypergraph vertices.
    A floor *geom* is added (a collision surface) but no body — so n_vertices is unchanged."""
    bare = _emit_arm_or_skip("data/robotics/anthropomorphic_arm.hymeko")
    pretty = beautify_mjcf(bare)
    m_bare = mujoco.MjModel.from_xml_string(bare)
    m_pretty = mujoco.MjModel.from_xml_string(pretty)
    assert m_pretty.nu == m_bare.nu
    assert m_pretty.njnt == m_bare.njnt
    assert m_pretty.nbody == m_bare.nbody          # floor is a geom, not a body
    assert (HypergraphState.from_mjcf(pretty, is_path=False).n_vertices
            == HypergraphState.from_mjcf(bare, is_path=False).n_vertices)
