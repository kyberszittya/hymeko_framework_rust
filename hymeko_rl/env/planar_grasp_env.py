"""Galambos planar two-finger grasping env — TOP-DOWN table; two connected planar arms pull a coin.

Galambos-sensei's "hello world" (docs/demo/galambos_scenario, local-only): two 2-link planar arms
(thumb + index) lie flat on a table and sweep in the XY plane (Z-axis hinges). A coin is **placed at
a random reachable spot on the table** (it does not fall); the arms pull it into a fixed zone between
them. Pure PPO; the policy reads the **kinematic hypergraph** of the two arms.

The arms are hand-authored MJCF (connected capsule links via ``fromto``, bases sized so the workspace
is well inside reach) — the emitter's geometry conventions cannot express connected planar rods, so
this is a hand-authored scene like ``arm_world``; the hypergraph is still derived from the MJCF. The
coin is a planar body (slide-x/slide-y/hinge-z) confined to the arms' plane; the zone is a marker site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import emit_arm_mjcf, with_collision_floor
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.hypergraph_state import HypergraphState

_PLANAR_ARM = "data/robotics/galambos_planar.hymeko"
_PLANAR_TASK = "data/robotics/galambos_task.hymeko"
_PLANE_Z = 0.04


def make_planar_arms_mjcf(*, base_x: float = 0.14, l1: float = 0.16, l2: float = 0.14,
                          z: float = _PLANE_Z) -> str:
    """A hand-authored two-arm planar scene (floor + two connected 2-link Z-hinge arms + position
    servos). Each arm: base hub → link1 (length ``l1``) → link2 (length ``l2``) → fingertip site,
    capsules connected ``fromto`` (no gaps). Bases at ``x = ±base_x``; at home both arms point +Y."""
    def arm(side: int, nm: str) -> str:
        return f'''<body name="base_{nm}" pos="{side * base_x:g} -0.02 {z:g}">
        <geom type="cylinder" size="0.022 0.012"/>
        <body name="link1_{nm}">
          <joint name="j1_{nm}" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
          <geom type="capsule" fromto="0 0 0 0 {l1:g} 0" size="0.012"/>
          <body name="link2_{nm}" pos="0 {l1:g} 0">
            <joint name="j2_{nm}" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
            <geom type="capsule" fromto="0 0 0 0 {l2:g} 0" size="0.01" rgba="0.4 0.6 0.95 1"/>
            <site name="tip_{nm}" pos="0 {l2:g} 0" size="0.013" rgba="0.9 0.7 0.1 1"/>
          </body>
        </body>
      </body>'''
    return f'''<mujoco model="galambos_planar">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/></visual>
  <default><joint damping="1.5"/><geom rgba="0.2 0.5 0.9 1" friction="1 0.05 0.001"/></default>
  <worldbody>
    <light pos="0 0.1 1.2" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.82 0.82 0.85 1"/>
    {arm(-1, "left")}
    {arm(1, "right")}
  </worldbody>
  <actuator>
    <position name="a_j1_left" joint="j1_left" kp="8" kv="1.0"/>
    <position name="a_j2_left" joint="j2_left" kp="8" kv="1.0"/>
    <position name="a_j1_right" joint="j1_right" kp="8" kv="1.0"/>
    <position name="a_j2_right" joint="j2_right" kp="8" kv="1.0"/>
  </actuator>
</mujoco>'''


def compose_planar_scene(arm_mjcf: str, *, disk_radius: float = 0.035, disk_half: float = 0.02,
                         zone_x: float = 0.0, zone_y: float = 0.16, zone_half: float = 0.055,
                         plane_z: float = _PLANE_Z) -> str:
    """Inject a planar table coin (slide-x/slide-y/hinge-z, confined to the arms' plane; placed, not
    dropped) + a target-zone marker site into a two-arm MJCF. Appended before ``</worldbody>``."""
    arm_mjcf = with_collision_floor(arm_mjcf, z=0.0)   # emitted arms carry no floor
    coin = (
        f'<body name="disk" pos="0 0 {plane_z:g}">'
        f'<joint name="disk_x" type="slide" axis="1 0 0" damping="2.5"/>'
        f'<joint name="disk_y" type="slide" axis="0 1 0" damping="2.5"/>'
        f'<joint name="disk_rz" type="hinge" axis="0 0 1" damping="0.8"/>'
        f'<geom name="disk" type="cylinder" size="{disk_radius:g} {disk_half:g}" '
        f'rgba="0.85 0.3 0.2 1" friction="1.0 0.05 0.001"/>'
        f'</body>'
    )
    zone = (
        f'<site name="target_zone" type="cylinder" size="{zone_half:g} 0.002" '
        f'pos="{zone_x:g} {zone_y:g} 0.004" rgba="0.2 0.8 0.3 0.4"/>'
    )
    return arm_mjcf.replace("</worldbody>", "    " + coin + zone + "\n  </worldbody>", 1)


@dataclass(frozen=True)
class PlanarGraspMetrics:
    """Live task state on the table: coin centre ``(x, y)``, in-plane distance to the zone, per-finger
    contact with the coin, and whether the coin is inside the zone."""

    disk_pos: np.ndarray
    disk_to_zone: float
    left_contact: bool
    right_contact: bool
    in_zone: bool


CLEAN_PLANAR = PlanarGraspMetrics(np.zeros(2, dtype=np.float32), 0.0, False, False, False)


def compute_planar_metrics(model: Any, data: Any, *, disk_body: int, disk_geom: int,
                           left_bodies: frozenset[int], right_bodies: frozenset[int],
                           zone_x: float, zone_y: float, zone_half: float) -> PlanarGraspMetrics:
    """Derive :class:`PlanarGraspMetrics` from a stepped ``(model, data)`` — all in the table plane."""
    pos = data.xpos[disk_body]
    disk_xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float32)
    to_zone = float(np.hypot(pos[0] - zone_x, pos[1] - zone_y))
    left = right = False
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 != disk_geom and g2 != disk_geom:
            continue
        ob = int(model.geom_bodyid[g2 if g1 == disk_geom else g1])
        if ob in left_bodies:
            left = True
        elif ob in right_bodies:
            right = True
    return PlanarGraspMetrics(disk_xy, to_zone, left, right, to_zone < zone_half)


class PlanarGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Top-down planar grasping: pull a coin placed on the table into the zone between two arms.

    Observation = per-vertex features on the two-arm hypergraph ``(n_vertices, 8)``: each link vertex
    carries its driving joint ``(qpos, qvel)``, its world position ``(x, y)``, its **vector to the
    coin** ``(coin - vertex)``, and the broadcast **coin→zone** vector. Action = 4 arm joint targets
    (the coin is unactuated). Reward = the declarative ``galambos_task.hymeko`` spec. The coin is
    placed in reach at reset (no fall). Ends on success (coin in zone for ``success_steps``), death
    (coin knocked out of the workspace), or ``max_steps``.
    """

    metadata = {"render_modes": []}
    _FEAT = 8

    def __init__(self, *, robot: str | None = _PLANAR_ARM, reward_spec: RewardSpec | None = None,
                 frame_skip: int = 5, max_steps: int = 160, zone_x: float = 0.0, zone_y: float = 0.16,
                 zone_half: float = 0.055, success_steps: int = 5, out_bound: float = 0.4) -> None:
        super().__init__()
        if frame_skip < 1 or max_steps < 1:
            raise ValueError("frame_skip/max_steps must be >= 1")
        # The robot is DESCRIBED IN HYMEKO (galambos_planar.hymeko) and emitted; `robot=None` falls
        # back to the hand-authored baseline (make_planar_arms_mjcf).
        arm_mjcf = (emit_arm_mjcf(robot, name="galambos", control_mode="position")
                    if robot is not None else make_planar_arms_mjcf())
        self.hg = HypergraphState.from_mjcf(arm_mjcf, is_path=False)
        mjcf = compose_planar_scene(arm_mjcf, zone_x=zone_x, zone_y=zone_y, zone_half=zone_half)
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.reward_spec = reward_spec if reward_spec is not None else \
            RewardSpec.from_hymeko(_PLANAR_TASK)
        self.frame_skip, self.max_steps = frame_skip, max_steps
        self.success_steps = success_steps
        self._zone_x, self._zone_y, self._zone_half = zone_x, zone_y, zone_half
        self._out_bound = out_bound

        self._disk_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "disk")
        self._disk_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                 for b in range(self.model.nbody)]
        self._left_bodies = frozenset(b for b, n in enumerate(names) if n and n.endswith("_left"))
        self._right_bodies = frozenset(b for b, n in enumerate(names) if n and n.endswith("_right"))
        self._arm_joints = [
            j for j in range(self.model.njnt)
            if not (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    or "").startswith("disk")]
        self._disk_x_adr = self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "disk_x")]
        self._disk_y_adr = self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "disk_y")]

        self.n_actions = int(self.model.nu)
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(self._ctrl_lo, self._ctrl_hi, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.hg.n_vertices, self._FEAT), dtype=np.float32)
        self._planar_metrics = CLEAN_PLANAR
        self._step = 0
        self._success = 0

    def _metrics(self) -> PlanarGraspMetrics:
        return compute_planar_metrics(
            self.model, self.data, disk_body=self._disk_body, disk_geom=self._disk_geom,
            left_bodies=self._left_bodies, right_bodies=self._right_bodies,
            zone_x=self._zone_x, zone_y=self._zone_y, zone_half=self._zone_half)

    def node_features(self) -> np.ndarray:
        feat = np.zeros((self.hg.n_vertices, self._FEAT), dtype=np.float32)
        for j in self._arm_joints:
            v = int(self.model.jnt_bodyid[j]) - 1
            if 0 <= v < self.hg.n_vertices:
                feat[v, 0] = self.data.qpos[self.model.jnt_qposadr[j]]
                feat[v, 1] = self.data.qvel[self.model.jnt_dofadr[j]]
        cx, cy = float(self._planar_metrics.disk_pos[0]), float(self._planar_metrics.disk_pos[1])
        for v in range(self.hg.n_vertices):
            vx, vy = float(self.data.xpos[v + 1][0]), float(self.data.xpos[v + 1][1])
            feat[v, 2], feat[v, 3] = vx, vy
            feat[v, 4], feat[v, 5] = cx - vx, cy - vy           # this link's vector to the coin
        feat[:, 6] = cx - self._zone_x                          # broadcast coin -> zone
        feat[:, 7] = cy - self._zone_y
        return feat

    def _sample_coin_xy(self) -> tuple[float, float]:
        """A reachable coin placement OUTSIDE the zone (so the arms must pull it in)."""
        for _ in range(64):
            x = float(self.np_random.uniform(-0.11, 0.11))
            y = float(self.np_random.uniform(0.08, 0.22))
            if np.hypot(x - self._zone_x, y - self._zone_y) >= self._zone_half + 0.03:
                return x, y
        return 0.10, 0.10

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0
        x, y = self._sample_coin_xy()
        self.data.qpos[self._disk_x_adr] = x
        self.data.qpos[self._disk_y_adr] = y
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._planar_metrics = self._metrics()
        self._step = 0
        self._success = 0
        return self.node_features(), {"disk_to_zone": self._planar_metrics.disk_to_zone}

    def step(self, action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ctrl = np.clip(np.asarray(action, dtype=np.float32), self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        self._planar_metrics = self._metrics()
        reward = self.reward_spec.evaluate(self, self._planar_metrics.disk_to_zone, ctrl)
        cx, cy = float(self._planar_metrics.disk_pos[0]), float(self._planar_metrics.disk_pos[1])
        death = abs(cx) > self._out_bound or cy < -0.08 or cy > 0.45
        self._success = self._success + 1 if self._planar_metrics.in_zone else 0
        terminated = bool(self._success >= self.success_steps or death)
        truncated = self._step >= self.max_steps
        return (self.node_features(), reward, terminated, truncated,
                {"disk_to_zone": self._planar_metrics.disk_to_zone,
                 "in_zone": self._planar_metrics.in_zone, "death": death,
                 "both_contact": self._planar_metrics.left_contact
                 and self._planar_metrics.right_contact})
