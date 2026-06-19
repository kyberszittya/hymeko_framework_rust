"""Verify a generated robot arm is *proper*: loads, articulates, reaches, and is connected.

Catches the robot-generation failure modes: links parked at the joint (stubs with gaps, not
connected rods), frozen joints, and degenerate / unreachable workspaces (the B-005 colinear-spinner).
Operates on an emitted MJCF + the end-effector body name — pair it with ``emit_arm_mjcf`` to
generate-and-verify a HyMeKo-described arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ArmReport:
    """Verification result for a generated arm."""

    loads: bool
    n_revolute: int
    articulates: bool       # every revolute joint moves the end-effector
    reach_radius: float     # max EE displacement from home over random joint configs (m)
    max_link_gap: float     # worst distance from a joint to its parent link's geometry (m)
    connected: bool         # every link attaches to its parent (max_link_gap < tol)

    @property
    def ok(self) -> bool:
        return (self.loads and self.articulates and self.connected
                and self.reach_radius > self._min_reach)

    _min_reach: float = 0.05

    def summary(self) -> str:
        flag = "OK" if self.ok else "NOT PROPER"
        return (f"[{flag}] revolute={self.n_revolute} articulates={self.articulates} "
                f"connected={self.connected} (max_gap={self.max_link_gap:.3f}m) "
                f"reach={self.reach_radius:.3f}m")


def verify_arm(mjcf: str, *, ee_body: str, gap_tol: float = 0.04, move_eps: float = 1e-3,
               n_samples: int = 200, seed: int = 0, min_reach: float = 0.05) -> ArmReport:
    """Load and probe an emitted arm MJCF; return an :class:`ArmReport`.

    # Preconditions ``mjcf`` is a MuJoCo XML string; ``ee_body`` names a body in it.
    # Postconditions ``loads`` is False (and the rest default) if the model fails to compile;
      otherwise the report reflects measured articulation / reach / connectivity.
    """
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
    except (ValueError, RuntimeError):
        return ArmReport(False, 0, False, 0.0, float("inf"), False, _min_reach=min_reach)
    data = mujoco.MjData(model)
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if ee < 0:
        return ArmReport(True, 0, False, 0.0, float("inf"), False, _min_reach=min_reach)
    rev = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    # EE point = the end-effector link's geom centroid (offset from the joint), not the body origin:
    # the leaf link's own joint pivots about the body origin, so that point would read as frozen.
    ee_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == ee]

    def ee_point() -> np.ndarray:
        return np.asarray(data.geom_xpos[ee_geoms[0]] if ee_geoms else data.xpos[ee])

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    home = ee_point().copy()
    home_geom = data.geom_xpos.copy()
    home_xmat = data.geom_xmat.copy()

    # articulation: each revolute joint, moved alone, must change SOME geom's pose — position OR
    # orientation. The orientation term catches roll joints (rotation about a link's own axis, where
    # an on-axis geom does not translate). Checked over all geoms (multi-arm safe).
    articulates = bool(rev)
    for j in rev:
        data.qpos[:] = 0.0
        data.qpos[model.jnt_qposadr[j]] = 0.5
        mujoco.mj_forward(model, data)
        moved = float(np.max(np.linalg.norm(data.geom_xpos - home_geom, axis=1)))
        rotated = float(np.max(np.abs(data.geom_xmat - home_xmat)))
        if max(moved, rotated) < move_eps:
            articulates = False

    # reach: max EE displacement from home over random joint configs.
    rng = np.random.default_rng(seed)
    reach = 0.0
    for _ in range(n_samples):
        data.qpos[:] = 0.0
        for j in rev:
            lo, hi = (model.jnt_range[j] if model.jnt_limited[j] else (-3.1416, 3.1416))
            data.qpos[model.jnt_qposadr[j]] = float(rng.uniform(lo, hi))
        mujoco.mj_forward(model, data)
        reach = max(reach, float(np.linalg.norm(ee_point() - home)))

    # connectivity: each revolute joint must sit within (≈) its parent link's geometry — a link
    # whose joint floats far from the parent's geom is a disconnected stub.
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    max_gap = 0.0
    for j in rev:
        parent = int(model.body_parentid[model.jnt_bodyid[j]])
        anchor = data.xanchor[j]
        pgeoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == parent]
        if not pgeoms:
            continue
        gap = min(max(0.0, float(np.linalg.norm(anchor - data.geom_xpos[g]))
                      - float(model.geom_rbound[g])) for g in pgeoms)
        max_gap = max(max_gap, gap)

    return ArmReport(True, len(rev), articulates, reach, max_gap, max_gap < gap_tol,
                     _min_reach=min_reach)
