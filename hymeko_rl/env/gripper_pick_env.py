"""Floating-gripper pick-and-place — the grasp+BC de-risk testbed (Phase 1, path B).

The gantry base *is* the palm position (slide x/y/z), so control is direct Cartesian: no IK, no
orientation, no reach envelope — the part the 6-DOF arm makes hard. The point is to prove the
expert -> behaviour-cloning -> learned-grasp loop here, then port it onto the arm (PickPlaceEnv) once
its IK/pedestal is sorted. Gripper from ``gripper.hymeko`` (emitted) or the hand-authored fallback; table +
freejoint box + target injected by ``compose_pick_place_scene``; the policy reads the gripper hypergraph.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import emit_arm_mjcf
from hymeko_rl.env.gripper_world import compose_pick_place_scene, make_gripper_mjcf
from hymeko_rl.hypergraph_state import HypergraphState

_GRIPPER = "data/robotics/gripper.hymeko"
_FEAT = 8
_BASE_Z = 0.22          # the gantry anchor height (gripper.hymeko slide_x world offset)
_GRIP = 0.03


class GripperPickEnv(gym.Env[np.ndarray, np.ndarray]):
    """Direct-Cartesian floating gripper. Obs ``(5, 8)`` per gripper-link vertex
    ``[qpos, qvel, x, y, z, (object - vertex)(3)]``; action = palm world target ``[x, y, z]`` + ``grip``."""

    metadata = {"render_modes": []}

    def __init__(self, *, robot: str | None = _GRIPPER, box_half: float = 0.02,
                 target_xy: tuple[float, float] = (0.12, 0.0), frame_skip: int = 5,
                 max_steps: int = 200, table_z: float = 0.0, grasp_z: float = 0.07,
                 lift_thresh: float = 0.06, place_radius: float = 0.05) -> None:
        super().__init__()
        if frame_skip < 1 or max_steps < 1:
            raise ValueError("frame_skip/max_steps must be >= 1")
        gripper = (emit_arm_mjcf(robot, name="gripper", control_mode="position")
                   if robot is not None else make_gripper_mjcf())
        self.hg = HypergraphState.from_mjcf(gripper, is_path=False)
        self._mjcf = compose_pick_place_scene(gripper, box_half=box_half, target_xy=target_xy,
                                              table_z=table_z)
        self.model = mujoco.MjModel.from_xml_string(self._mjcf)
        self.model.actuator_ctrllimited[:] = 0
        self.data = mujoco.MjData(self.model)
        self.frame_skip, self.max_steps = int(frame_skip), int(max_steps)
        self.box_half, self.table_z, self.grasp_z = box_half, float(table_z), float(grasp_z)
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))
        self.lift_thresh, self.place_radius = float(lift_thresh), float(place_radius)

        def bid(n: str) -> int:
            return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n))

        self._b_obj, self._b_palm = bid("object"), bid("palm")
        self._b_fl, self._b_fr = bid("finger_l"), bid("finger_r")
        self._obj_qadr = int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj")])
        self._act_of: dict[str, int] = {}
        for a in range(self.model.nu):
            j = int(self.model.actuator_trnid[a, 0])
            self._act_of[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""] = a
        # stiff, controllable gains (the emitter defaults are too soft); disable gantry self-collision.
        for jn, (kp, kv) in {"slide_x": (120.0, 8.0), "slide_y": (120.0, 8.0), "slide_z": (200.0, 12.0),
                             "grip_l": (90.0, 4.0), "grip_r": (90.0, 4.0)}.items():
            if jn in self._act_of:
                a = self._act_of[jn]
                self.model.actuator_gainprm[a, 0] = kp
                self.model.actuator_biasprm[a, 1] = -kp
                self.model.actuator_biasprm[a, 2] = -kv
        for nm in ("carriage_x", "carriage_y", "palm"):
            b = bid(nm)
            if b >= 0:
                for g in range(int(self.model.body_geomadr[b]),
                               int(self.model.body_geomadr[b]) + int(self.model.body_geomnum[b])):
                    self.model.geom_contype[g] = 0
                    self.model.geom_conaffinity[g] = 0
        self._jnt: list[tuple[int, int, int]] = []
        for jn in ("slide_x", "slide_y", "slide_z", "grip_l", "grip_r"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                v = int(self.model.jnt_bodyid[jid]) - 1
                self._jnt.append((v, int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])))

        self._a_lo = np.array([-0.25, -0.25, table_z + 0.03, 0.0], np.float32)
        self._a_hi = np.array([0.25, 0.25, _BASE_Z + 0.02, _GRIP], np.float32)
        self.action_space = spaces.Box(self._a_lo, self._a_hi)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.hg.n_vertices, _FEAT), np.float32)
        self.n_actions = 4
        self._step = 0

    def _obj_xyz(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._b_obj], dtype=np.float32)

    def _both_contact(self) -> tuple[bool, bool]:
        left = right = False
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[int(con.geom1)])
            b2 = int(self.model.geom_bodyid[int(con.geom2)])
            if self._b_obj not in (b1, b2):
                continue
            other = b2 if b1 == self._b_obj else b1
            if other == self._b_fl:
                left = True
            elif other == self._b_fr:
                right = True
        return left, right

    def node_features(self) -> np.ndarray:
        feat = np.zeros((self.hg.n_vertices, _FEAT), dtype=np.float32)
        ox, oy, oz = (float(c) for c in self._obj_xyz())
        for v, qadr, dadr in self._jnt:
            if 0 <= v < self.hg.n_vertices:
                feat[v, 0] = float(self.data.qpos[qadr])
                feat[v, 1] = float(self.data.qvel[dadr])
        for v in range(self.hg.n_vertices):
            x, y, z = (float(c) for c in self.data.xpos[v + 1])
            feat[v, 2:5] = (x, y, z)
            feat[v, 5:8] = (ox - x, oy - y, oz - z)
        return feat

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        bx = float(self.np_random.uniform(-0.12, 0.12))
        by = float(self.np_random.uniform(-0.12, 0.12))
        q = self._obj_qadr
        self.data.qpos[q:q + 3] = [bx, by, self.table_z + self.box_half]
        self.data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self.node_features(), {"obj_to_target": self._obj_to_target()}

    def _obj_to_target(self) -> float:
        o = self._obj_xyz()
        return float(np.hypot(o[0] - self.target_xy[0], o[1] - self.target_xy[1]))

    def _apply(self, palm_xyz_grip: np.ndarray) -> None:
        x, y, z, grip = (float(v) for v in palm_xyz_grip)
        self.data.ctrl[self._act_of["slide_x"]] = x
        self.data.ctrl[self._act_of["slide_y"]] = y
        self.data.ctrl[self._act_of["slide_z"]] = z - _BASE_Z        # palm world-z -> slide_z target
        self.data.ctrl[self._act_of["grip_l"]] = grip
        self.data.ctrl[self._act_of["grip_r"]] = -grip

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float32), self._a_lo, self._a_hi)
        self._apply(a)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        o = self._obj_xyz()
        palm = np.asarray(self.data.xpos[self._b_palm], dtype=np.float32)
        approach = float(np.linalg.norm(palm - o))
        left, right = self._both_contact()
        lifted = float(max(0.0, o[2] - (self.table_z + self.box_half)))
        to_target = self._obj_to_target()
        reached = bool(lifted > self.lift_thresh and to_target < self.place_radius)
        reward = (-approach + 0.5 * (left + right) + 5.0 * min(lifted, self.lift_thresh)
                  + (-to_target if lifted > 0.02 else 0.0) + (20.0 if reached else 0.0))
        death = bool(o[2] < self.table_z - 0.05 or abs(o[0]) > 0.6 or abs(o[1]) > 0.6)
        terminated = bool(reached or death)
        truncated = self._step >= self.max_steps
        info = {"obj_to_target": to_target, "lifted": lifted, "both_contact": left and right,
                "reached": reached, "death": death}
        return self.node_features(), float(reward), terminated, truncated, info

    @property
    def expert_action(self) -> np.ndarray:
        """Trivial direct-Cartesian demonstrator (the point of path B): move the palm above the box,
        descend, close, lift, carry, place. Single-valued function of state (phase read from contact/height)."""
        o = self._obj_xyz()
        palm = np.asarray(self.data.xpos[self._b_palm], dtype=np.float32)
        grasped = all(self._both_contact())
        tx, ty = self.target_xy
        high_z, gz = _BASE_Z - 0.02, self.grasp_z
        near_place = float(np.hypot(o[0] - tx, o[1] - ty)) < 0.04
        horiz = float(np.hypot(palm[0] - o[0], palm[1] - o[1]))
        if grasped and not near_place:                       # carry to target (held high)
            tgt = np.array([tx, ty, high_z, _GRIP], np.float32)
        elif grasped and near_place:                         # lower then release
            below = palm[2] < gz + 0.03
            tgt = np.array([tx, ty, gz, 0.0 if below else _GRIP], np.float32)
        elif horiz > 0.02:                                   # move above the box, open
            tgt = np.array([o[0], o[1], high_z, 0.0], np.float32)
        elif palm[2] > gz + 0.02:                            # descend onto the box, open
            tgt = np.array([o[0], o[1], gz, 0.0], np.float32)
        else:                                                # at the box: close
            tgt = np.array([o[0], o[1], gz, _GRIP], np.float32)
        return tgt
