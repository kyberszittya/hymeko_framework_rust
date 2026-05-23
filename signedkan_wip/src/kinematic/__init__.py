"""Kinematic-graph utilities — robot pose, MuJoCo bridge, kinematic
fixtures.

Moved here from ``signedkan_wip/src/`` flat layout on 2026-05-19 as
part of Slice G-1 of the directory-reorganisation
(``docs/plans/2026-05-19-signedkan-wip-organize/``). The original
flat module names (``kinematic_fixtures.py``, ``kinematic_graph.py``,
``mujoco_bridge.py``, ``render_mujoco_video.py``) become topical
submodules under this package, and the public API is re-exported
flat through this ``__init__`` so external callers don't change.

Submodules:

* :mod:`.fixtures` — pre-built robot test rigs (canonical 7-DOF arm,
  planar 2R, etc).
* :mod:`.graph` — the ``KinematicGraph`` class + sign extraction
  helpers.
* :mod:`.mujoco_bridge` — MuJoCo XML / qpos / xfrmat plumbing
  (``MuJoCoBridge`` class).
* :mod:`.render_mujoco_video` — utility that wraps a trained
  classifier into a side-by-side MuJoCo replay (``main``-style CLI).

OO commitment: ``KinematicGraph`` is the canonical dataclass exported
here; ``MuJoCoBridge`` is the simulator-side wrapper.
"""
from .fixtures import _serial_arm_urdf, write_fixture
from .graph import (
    JOINT_SIGN,
    KinematicJoint,
    parse_urdf,
    urdf_to_signed_graph,
    kinematic_loop_summary,
)
from .mujoco_bridge import MuJoCoBridge
from .render_mujoco_video import render_video

__all__ = [
    "_serial_arm_urdf",
    "write_fixture",
    "JOINT_SIGN",
    "KinematicJoint",
    "parse_urdf",
    "urdf_to_signed_graph",
    "kinematic_loop_summary",
    "MuJoCoBridge",
    "render_video",
]
