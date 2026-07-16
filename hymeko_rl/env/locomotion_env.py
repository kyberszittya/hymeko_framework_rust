"""Fast-dynamics locomotion envs on HyMeKo-declared plants — the substrate layer of the fast-control
demonstration. A shared :class:`LocomotionEnv` base holds the MuJoCo machinery and the (duck-typed) reward
contract shared with :class:`~hymeko_rl.env.quadruped_env.QuadrupedGoalEnv`; two subclasses differ only by
base promotion, task features, termination, and scripted expert:

  * :class:`LeggedLocomotionEnv`  — torque-driven legged runner (``half_cheetah.hymeko`` planar,
                                    ``humanoid.hymeko`` free base). Task: run forward fast; fall terminates.
  * :class:`WheeledVehicleEnv`    — torque-driven diff-drive vehicle (``robot_4wh.hymeko``, freejoint base).
                                    Task: follow a high-speed waypoint track; rollover terminates.

Factored, not triplicated (§6.5 #3): the emitter/floor/base helpers are reused from
:mod:`hymeko_rl.env.arm_world` + :func:`hymeko_rl.env.quadruped_env.set_base_mode`; rewards are the shared
declarative :class:`hymeko_rl.env.reward.RewardSpec`; observations are the ``(N, 2)`` per-vertex hypergraph
tensor every HSiKAN testbed speaks. The plants live in ``data/robotics/`` and are unchanged."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.env.arm_world import emit_arm_mjcf, with_collision_floor
from hymeko_rl.env.locomotion_experts import AckermannPursuit, CpgGait, DiffDrivePursuit
from hymeko_rl.env.quadruped_env import set_base_mode
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.env.track_gen import race_circuit

_REPO = Path(__file__).resolve().parents[2]
_ROBOTICS = _REPO / "data" / "robotics"

# Run forward as fast as possible, smoothly, without falling. `goal_progress` toward a far goal telescopes to
# forward displacement per step (the dense speed driver); `alive` is the healthy bonus (fall terminates, so it
# is only collected while upright); action/jerk costs refine. reward.argmax = fast upright forward motion.
RUN_REWARD = RewardSpec((
    ("goal_progress", 60.0),        # +(prev_dist − dist) = forward displacement/step → the speed driver
    ("alive", 1.0),                 # +1/step while not fallen (fall terminates → survive to keep it)
    ("action_cost", 0.05),          # −‖a‖²: control effort
    ("joint_acceleration", 0.002),  # −Σ(Δq̇)²: smoothness / anti-jerk
))

# Reach the active waypoint fast, then the next. `goal_progress` toward the waypoint + a `success_bonus` on
# arrival + time pressure = drive fast along the track; rollover terminates (guarded outside the reward).
VEHICLE_REWARD = RewardSpec((
    ("goal_progress", 40.0),        # +(prev_dist − dist): progress toward the active waypoint
    ("success_bonus", 25.0),        # +25 on the step a waypoint is reached
    ("action_cost", 0.02),          # −‖a‖²: wheel-torque effort
    ("time_penalty", 0.05),         # −0.05/step: reach waypoints sooner
))


def circular_track(cx: float, cy: float, radius: float, n: int, start_deg: float = -45.0) -> np.ndarray:
    """``n`` waypoints evenly spaced CCW around a circle — a smooth racing loop a diff-drive can hold at speed
    (dense waypoints keep the per-segment heading change small, so pure-pursuit does not chord-cut). # Returns
    ``(n, 2)`` float64 xy waypoints."""
    th = np.deg2rad(start_deg) + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1)


def straight_track(first: float, spacing: float, n: int) -> np.ndarray:
    """``n`` waypoints along +x (a high-speed straight/drag course): the car accelerates toward a receding line
    of gates, so ``goal_progress`` rewards forward speed and the sprint does not terminate early. # Returns
    ``(n, 2)`` float64 xy waypoints."""
    return np.stack([first + spacing * np.arange(n, dtype=np.float64), np.zeros(n)], axis=1)


@dataclass(frozen=True)
class LocomotionConfig:
    """Declarative build config for a locomotion env. The plant geometry is the ``.hymeko``; this carries the
    simulation tuning + task the geometry cannot. # Preconditions ``frame_skip >= 1``, ``max_steps >= 1``,
    ``ctrl_range > 0``. # Invariants none of these mutate after construction (frozen)."""

    plant: str                              # .hymeko plant path (absolute or under data/robotics/)
    name: str                               # short substrate id (cheetah / humanoid / vehicle)
    root_body: str                          # the torso/chassis body (uprightness + height reference)
    base_joint: str = "base"                # the declared base joint promoted to free/planar (legged)
    base_mode: str = "free"                 # "free" | "planar" (legged); vehicle injects a freejoint
    ctrl_range: float = 80.0                # |torque| per actuator at action = ±1
    frame_skip: int = 5
    max_steps: int = 300
    init_noise: float = 0.01
    joint_damping: float = 1.0
    joint_armature: float = 0.02
    fall_height: float = 0.25               # torso z below which the body has fallen (legged)
    flip_cos: float = 0.0                   # uprightness below which the torso has tipped over
    goal_distance: float = 1000.0           # far forward goal (legged) — makes goal_progress ≈ speed
    reach_radius: float = 0.5               # waypoint arrival radius (vehicle)
    reward_spec: RewardSpec | None = None
    terrain: str | None = None              # None/"flat" → flat floor; "hills"/"bumps"/"ramps" → heightfield
    terrain_relief: float = 0.15            # heightfield peak height [m] (keep low; big relief flips vehicles)
    terrain_radius: float = 16.0            # heightfield half-extent [m]
    terrain_seed: int = 0

    def resolve_plant(self) -> str:
        p = Path(self.plant)
        return str(p if p.is_absolute() else _ROBOTICS / self.plant)


class LocomotionEnv(gym.Env[np.ndarray, np.ndarray]):
    """Base MuJoCo locomotion env on a HyMeKo plant. Owns the model/data, the ``(N, 2)`` per-vertex
    hypergraph observation, the shared step/reset, and the duck-typed reward attributes the
    :class:`RewardSpec` terms read (``_prev_dist``, ``_act_dofs``, ``_prev_act_qvel``, ``_stand_height``,
    ``torso``, ``reach_thresh``, ``_reached``). Subclasses implement base promotion, the torso-vertex task
    feature, task advancement, termination, and the scripted expert.

    # Preconditions a valid :class:`LocomotionConfig`. # Invariants ``observation`` is ``(n_vertices, 2)``
    float32; ``action`` is ``n_actions`` torques normalised to ``[-1, 1]`` (scaled to ``±ctrl_range``).
    # Postconditions ``step`` returns the gym 5-tuple; ``terminated`` per the subclass predicate; ``reward``
    the configured :class:`RewardSpec`."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: LocomotionConfig) -> None:
        super().__init__()
        if cfg.frame_skip < 1 or cfg.max_steps < 1 or cfg.ctrl_range <= 0:
            raise ValueError("frame_skip/max_steps >= 1 and ctrl_range > 0 required")
        self.cfg = cfg
        raw = emit_arm_mjcf(cfg.resolve_plant(), name=cfg.name)
        mjcf = self._prepare_mjcf(raw)                       # subclass: promote the base
        self._terrain_elev: np.ndarray | None = None
        if cfg.terrain and cfg.terrain != "flat":
            mjcf = self._apply_terrain(mjcf)
        else:
            mjcf = self._seat_floor(mjcf)
        self._mjcf = mjcf
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        if self._terrain_elev is not None:
            from hymeko_rl.env.terrain import fill_hfield
            fill_hfield(self.model, self._terrain_elev)
        self.hg = HypergraphState.from_mjcf(mjcf, is_path=False)
        self.torso = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.root_body))
        if self.torso < 0:
            raise ValueError(f"root body {cfg.root_body!r} not found in the emitted plant")
        self._index_actuators()
        self._tune_dofs(cfg.joint_damping, cfg.joint_armature)
        self.ctrl_range = float(cfg.ctrl_range)
        self.frame_skip = int(cfg.frame_skip)
        self.max_steps = int(cfg.max_steps)
        self.init_noise = float(cfg.init_noise)
        self._dt = float(self.model.opt.timestep)
        self.reach_thresh = float(cfg.reach_radius)
        self.reward_spec = cfg.reward_spec if cfg.reward_spec is not None else self._default_reward()
        mujoco.mj_forward(self.model, self.data)
        self._q0 = self.data.qpos.copy()
        self._stand_height = float(self.data.xpos[self.torso, 2])
        self._torso_vtx = self.torso - 1
        self._jnt_vtx = np.array([int(self.model.jnt_bodyid[j]) - 1 for j in range(self.model.njnt)],
                                 dtype=np.int64)
        self._jnt_qadr = self.model.jnt_qposadr.astype(np.int64)
        self._jnt_dadr = self.model.jnt_dofadr.astype(np.int64)
        # z qpos address of the free base (for the terrain spawn-lift): first free joint's qpos + 2.
        self._base_z_qadr = next((int(self.model.jnt_qposadr[j]) + 2 for j in range(self.model.njnt)
                                  if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE), None)
        self._reset_task_state()
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.hg.n_vertices, 2), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actions,), np.float32)

    # ── subclass hooks ───────────────────────────────────────────────────────
    def _prepare_mjcf(self, raw_mjcf: str) -> str:
        raise NotImplementedError

    def _default_reward(self) -> RewardSpec:
        raise NotImplementedError

    def _root_features(self) -> tuple[float, float]:
        """The two scalars carried on the torso/chassis vertex (task-relevant state)."""
        raise NotImplementedError

    def _advance_task(self, dist: float) -> None:
        """Post-reward task bookkeeping (e.g. waypoint advance). Default: nothing."""

    def _is_terminated(self, upright: float) -> bool:
        raise NotImplementedError

    def dist_to_goal(self) -> float:
        raise NotImplementedError

    @property
    def expert_action(self) -> np.ndarray:
        raise NotImplementedError

    def _action_dim(self) -> int:
        """Action-space dimension. Default: one action per actuator (torque/velocity envs). Ackermann overrides
        it to a 2-D ``[throttle, steer]`` interface that fans out to its 4 actuators in :meth:`_apply_control`."""
        return int(self.model.nu)

    def _apply_control(self, a: np.ndarray) -> None:
        """Map the normalised action ``a ∈ [-1, 1]^n_actions`` to ``data.ctrl``. Default: uniform scale to
        ``±ctrl_range`` over every actuator (torque, or wheel-velocity target for the diff-drive envs)."""
        self.data.ctrl[self._ctrl_idx] = a * self.ctrl_range

    # ── shared machinery ─────────────────────────────────────────────────────
    def _index_actuators(self) -> None:
        """Index the plant's actuators (the base motor was stripped by base promotion) and set the action-space
        dimension via :meth:`_action_dim`. # Postconditions ``_ctrl_idx`` covers all actuators; ``_act_dofs``/
        ``_act_qadr`` are the actuated joints' dof/qpos addresses; ``n_actions == _action_dim()``."""
        self._ctrl_idx = np.arange(int(self.model.nu), dtype=np.int64)
        if int(self.model.nu) == 0:
            raise ValueError("no actuators on the emitted plant — is the base motor the only one?")
        jnts = [int(self.model.actuator_trnid[i, 0]) for i in self._ctrl_idx]
        self._act_dofs = np.array([int(self.model.jnt_dofadr[j]) for j in jnts], dtype=np.int64)
        self._act_qadr = np.array([int(self.model.jnt_qposadr[j]) for j in jnts], dtype=np.int64)
        self.n_actions = self._action_dim()

    def _tune_dofs(self, damping: float, armature: float) -> None:
        """Add damping + reflected inertia to the ACTUATED dofs only (not the passive base), precisely, on the
        built model — avoids the MJCF-regex fragility of matching hinge names across three plants."""
        for d in self._act_dofs:
            self.model.dof_damping[int(d)] = float(damping)
            self.model.dof_armature[int(d)] = float(armature)

    @staticmethod
    def _seat_z(mjcf: str) -> float:
        """The z just under the lowest geom at the rest pose — where the ground should sit."""
        m = mujoco.MjModel.from_xml_string(mjcf)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        bottoms = [float(d.geom_xpos[g, 2] - m.geom_rbound[g]) for g in range(m.ngeom)
                   if int(m.geom_bodyid[g])]
        return (min(bottoms) - 0.01) if bottoms else -0.01

    @staticmethod
    def _seat_floor(mjcf: str) -> str:
        """Place a flat collision floor just under the lowest geom at the rest pose (reused from the quadruped)."""
        return with_collision_floor(mjcf, z=LocomotionEnv._seat_z(mjcf))

    def _apply_terrain(self, mjcf: str) -> str:
        """Swap the flat floor for a procedural heightfield (`env.terrain`). Stores the elevation grid for
        :func:`terrain.fill_hfield` after the model is built. Uses an earthy rgba (the env physics model has no
        beautify material)."""
        from hymeko_rl.env.terrain import procedural_hfield, with_hfield
        self._terrain_elev = procedural_hfield(self.cfg.terrain, seed=self.cfg.terrain_seed)
        return with_hfield(mjcf, radius=self.cfg.terrain_radius, z_top=self.cfg.terrain_relief,
                           z_pos=self._seat_z(mjcf), material=None)

    def _torso_uprightness(self) -> float:
        """cos between the torso local +z and world +z (1 = level, 0 = on its side, <0 = inverted)."""
        return float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])

    def node_features(self) -> np.ndarray:
        """Per-vertex ``(N, 2)`` obs: each actuated-joint vertex carries ``[qpos, qvel]``; the torso/chassis
        vertex carries the subclass task features. Passive base joints (mapped to the torso vertex) are
        skipped. # Postconditions shape ``(hg.n_vertices, 2)`` float32."""
        feat = np.zeros((self.hg.n_vertices, 2), dtype=np.float32)
        for j in range(self.model.njnt):
            v = int(self._jnt_vtx[j])
            if v == self._torso_vtx or not 0 <= v < self.hg.n_vertices:
                continue
            feat[v, 0] = float(self.data.qpos[int(self._jnt_qadr[j])])
            feat[v, 1] = float(self.data.qvel[int(self._jnt_dadr[j])])
        f0, f1 = self._root_features()
        feat[self._torso_vtx, 0] = f0
        feat[self._torso_vtx, 1] = f1
        return feat

    def _reset_task_state(self) -> None:
        self._step = 0
        self._prev_x = float(self.data.xpos[self.torso, 0])
        self._prev_y = float(self.data.xpos[self.torso, 1])
        self._vx = 0.0
        self._reached = False
        self._prev_dist = self.dist_to_goal()
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = self._q0
        self.data.qpos[:] += self.np_random.uniform(-self.init_noise, self.init_noise, self.model.nq)
        self.data.qvel[:] = 0.0
        if self._terrain_elev is not None and self._base_z_qadr is not None:
            self.data.qpos[self._base_z_qadr] += self.cfg.terrain_relief + 0.15   # spawn above → settle onto terrain
        mujoco.mj_forward(self.model, self.data)
        self._reset_task_state()
        return self.node_features(), {"dist": self.dist_to_goal()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._apply_control(a)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        x = float(self.data.xpos[self.torso, 0])
        self._vx = (x - self._prev_x) / (self.frame_skip * self._dt)
        self._prev_x = x
        dist = self.dist_to_goal()
        self._reached = bool(dist < self.reach_thresh)
        upright = self._torso_uprightness()
        reward = float(self.reward_spec.evaluate(self, dist, a))
        self._prev_dist = dist
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()
        self._advance_task(dist)                              # e.g. waypoint advance (resets _prev_dist)
        terminated = self._is_terminated(upright)
        truncated = self._step >= self.max_steps
        info = {"dist": dist, "vx": self._vx, "upright": upright, "x": x,
                "reached": self._reached, "fallen": terminated, "step": self._step}
        return self.node_features(), reward, terminated, truncated, info

    def privileged_state(self) -> np.ndarray:
        """Full flat privileged state (qpos ⊕ qvel ⊕ torso xpos) for an asymmetric-CTDE critic (§ optional
        trainer feature). # Postconditions 1-D float32."""
        return np.concatenate([self.data.qpos, self.data.qvel,
                               self.data.xpos[self.torso]]).astype(np.float32)


class LeggedLocomotionEnv(LocomotionEnv):
    """Torque-driven legged runner (cheetah planar / humanoid free base). Runs toward a far forward goal;
    a fall (torso below ``fall_height`` OR tipped past ``flip_cos``) terminates. Scripted expert: a bounding
    :class:`CpgGait`."""

    def __init__(self, cfg: LocomotionConfig, *, gait_freq: float = 1.6, gait_amp: float = 0.7) -> None:
        super().__init__(cfg)
        self.goal_x = float(self.data.xpos[self.torso, 0]) + float(cfg.goal_distance)
        self._gait = CpgGait.alternating(self.n_actions, freq=gait_freq, amp=gait_amp)

    def _prepare_mjcf(self, raw_mjcf: str) -> str:
        return set_base_mode(raw_mjcf, self.cfg.base_joint, self.cfg.base_mode)

    def _default_reward(self) -> RewardSpec:
        return RUN_REWARD

    def dist_to_goal(self) -> float:
        return float(self.goal_x - self.data.xpos[self.torso, 0]) if hasattr(self, "goal_x") \
            else float(self.cfg.goal_distance)

    def _root_features(self) -> tuple[float, float]:
        return float(self.dist_to_goal()), float(self._vx)     # [distance ahead, forward speed]

    def _is_terminated(self, upright: float) -> bool:
        z = float(self.data.xpos[self.torso, 2])
        return bool(z < self.cfg.fall_height or upright < self.cfg.flip_cos)

    @property
    def expert_action(self) -> np.ndarray:
        return self._gait.action(self)


class WheeledVehicleEnv(LocomotionEnv):
    """Torque-driven diff-drive vehicle (``robot_4wh``). Follows a high-speed waypoint track; a rollover
    (uprightness below ``flip_cos``) or completing the track terminates. Scripted expert: a pure-pursuit
    :class:`DiffDrivePursuit`."""

    # A wide circular racing loop (centre (0, 5) m, radius 5 m), 16 waypoints CCW starting ahead of the
    # vehicle's spawn (origin, facing +x). Continuous gentle curvature a diff-drive holds at speed — a car
    # doing laps. Dense waypoints keep each heading step small so pure-pursuit tracks smoothly, not chord-cuts.
    _DEFAULT_TRACK = circular_track(0.0, 5.0, 5.0, 16)
    _KV = 12.0                              # velocity-servo feedback gain (wheel ω tracking)

    def __init__(self, cfg: LocomotionConfig, *, track: np.ndarray | None = None,
                 pursuit: DiffDrivePursuit | None = None) -> None:
        self._track = np.asarray(track if track is not None else self._DEFAULT_TRACK, dtype=np.float64)
        self._wp = 0
        super().__init__(cfg)
        self._pursuit = pursuit if pursuit is not None else DiffDrivePursuit()

    def _prepare_mjcf(self, raw_mjcf: str) -> str:
        """Two rewrites on the emitted ``robot_4wh`` plant:

        1. It declares no ``@base`` joint (the chassis is the root), so inject a ``<freejoint>`` into the chassis
           body to free it — the vehicle analogue of :func:`set_base_mode`.
        2. The chassis controller is declared *velocity* in the ``.hymeko`` but the emitter produced torque
           ``<motor>`` wheels; unbounded wheel torque spins the light wheels to runaway speed (no traction) and
           the reaction torque flips the chassis. Convert the wheel motors to ``<velocity>`` servos so a
           commanded wheel angular velocity (``action · ctrl_range``) is tracked — the honest realisation of the
           declared diff-drive velocity interface."""
        body = self.cfg.root_body
        pattern = rf'(<body name="{re.escape(body)}"[^>]*>)'
        if not re.search(pattern, raw_mjcf):
            raise ValueError(f"chassis body {body!r} not found in the emitted vehicle plant")
        mjcf = re.sub(pattern, rf'\1<freejoint name="{self.cfg.base_joint}"/>', raw_mjcf, count=1)
        mjcf, n = re.subn(r'<motor name="(act_[^"]+)" joint="([^"]+)"[^>]*/>',
                          rf'<velocity name="\1" joint="\2" kv="{self._KV}"/>', mjcf)
        if n == 0:
            raise ValueError("expected wheel <motor> actuators to convert to <velocity> servos")
        return mjcf

    def _default_reward(self) -> RewardSpec:
        return VEHICLE_REWARD

    def _waypoint(self) -> np.ndarray:
        return self._track[min(self._wp, len(self._track) - 1)]

    def dist_to_goal(self) -> float:
        wp = self._waypoint()
        return float(np.hypot(self.data.xpos[self.torso, 0] - wp[0], self.data.xpos[self.torso, 1] - wp[1]))

    def heading_error(self) -> float:
        """Signed angle (radians, + = waypoint to the vehicle's left) between the chassis forward (+x local)
        and the direction to the active waypoint — the pure-pursuit steering signal."""
        wp = self._waypoint()
        to_wp = np.array([wp[0] - self.data.xpos[self.torso, 0], wp[1] - self.data.xpos[self.torso, 1]])
        fwd = self.data.xmat[self.torso].reshape(3, 3)[:2, 0]     # chassis local +x in world (planar)
        ang = np.arctan2(to_wp[1], to_wp[0]) - np.arctan2(fwd[1], fwd[0])
        return float(np.arctan2(np.sin(ang), np.cos(ang)))       # wrap to (−π, π]

    def _root_features(self) -> tuple[float, float]:
        return float(self.dist_to_goal()), float(self.heading_error())

    def _advance_task(self, dist: float) -> None:
        if dist < self.reach_thresh and self._wp < len(self._track) - 1:
            self._wp += 1
            self._prev_dist = self.dist_to_goal()                # rebaseline progress to the new waypoint
            self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()

    def _is_terminated(self, upright: float) -> bool:
        completed = bool(self._wp >= len(self._track) - 1 and self.dist_to_goal() < self.reach_thresh)
        return bool(upright < self.cfg.flip_cos or completed)    # rollover or track completed

    def _reset_task_state(self) -> None:
        self._wp = 0
        super()._reset_task_state()

    @property
    def expert_action(self) -> np.ndarray:
        return self._pursuit.action(self)


class AckermannCarEnv(WheeledVehicleEnv):
    """Ackermann-steered car (F1TENTH): the two front wheels steer, the two rear wheels drive. Reuses the
    waypoint-track task from :class:`WheeledVehicleEnv` but replaces the diff-drive control with the F1TENTH
    **2-D ``[throttle, steer]``** interface: ``throttle``→rear wheel velocity, ``steer``→front steer angle.

    # Invariants ``action`` is ``[throttle, steer] ∈ [-1, 1]²``; ``throttle·ctrl_range`` = rear wheel ω,
    ``steer·max_steer`` = front steer angle (rad). # Postconditions steering the front wheels toward a +y
    waypoint (``heading_error > 0``) turns the car toward it."""

    _KV = 8.0                               # rear velocity-servo gain (low: the 0.2 kg wheels are light)
    _KP = 8.0                               # front steer position-servo gain
    _DRIVE_FORCE = 1.0                      # rear drive torque limit [N·m]: caps the standing-start transient
    #                                         that would otherwise wheelie the light chassis (real motors are
    #                                         torque-limited); yields a stable clean-lap F1TENTH pace (~2.5 m/s).

    def __init__(self, cfg: LocomotionConfig, *, track: np.ndarray | None = None,
                 max_steer: float = 0.4, pursuit: AckermannPursuit | None = None) -> None:
        self._max_steer = float(max_steer)
        super().__init__(cfg, track=track)
        self._pursuit = pursuit if pursuit is not None else AckermannPursuit()
        # ctrl indices of the two rear drives and the two front steers (by actuator name, after the rewrite).
        self._rear_ctrl = self._named_ctrl(("act_rl", "act_rr"))
        self._steer_ctrl = self._named_ctrl(("act_steer_fl", "act_steer_fr"))

    def _tune_dofs(self, damping: float, armature: float) -> None:
        """Tune EVERY non-free joint dof — not just the actuated ones (the base default) — because the
        free-rolling front wheels also need reflected inertia. Light 0.2 kg wheels under a velocity/position
        servo blow up (QACC NaN) without enough armature; ``max(armature, 0.02)`` is the stability floor."""
        arm = max(float(armature), 0.02)
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            d = int(self.model.jnt_dofadr[j])
            self.model.dof_damping[d] = float(damping)
            self.model.dof_armature[d] = arm

    def _named_ctrl(self, names: tuple) -> np.ndarray:
        idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
        if any(i < 0 for i in idx):
            raise ValueError(f"expected actuators {names} on the emitted F1TENTH; got a missing one ({idx})")
        return np.asarray(idx, dtype=np.int64)

    def _prepare_mjcf(self, raw_mjcf: str) -> str:
        """Free the chassis, then realise the Ackermann drivetrain from the emitted torque motors: rear
        wheels → ``<velocity>`` servos (drive), front steer hinges → ``<position>`` servos (angle), and the
        free-rolling front wheels → no actuator (stripped)."""
        body = self.cfg.root_body
        pattern = rf'(<body name="{re.escape(body)}"[^>]*>)'
        if not re.search(pattern, raw_mjcf):
            raise ValueError(f"chassis body {body!r} not found in the emitted F1TENTH plant")
        mjcf = re.sub(pattern, rf'\1<freejoint name="{self.cfg.base_joint}"/>', raw_mjcf, count=1)
        mjcf = re.sub(r'\s*<motor name="act_roll_f[lr]"[^>]*/>', "", mjcf)          # free-rolling front wheels
        mjcf, nr = re.subn(r'<motor name="(act_r[lr])" joint="([^"]+)"[^>]*/>',
                           rf'<velocity name="\1" joint="\2" kv="{self._KV}" '
                           rf'forcerange="{-self._DRIVE_FORCE} {self._DRIVE_FORCE}"/>', mjcf)
        mjcf, ns = re.subn(r'<motor name="(act_steer_f[lr])" joint="([^"]+)"[^>]*/>',
                           rf'<position name="\1" joint="\2" kp="{self._KP}"/>', mjcf)
        if nr != 2 or ns != 2:
            raise ValueError(f"F1TENTH actuator rewrite mismatch (rear={nr}, steer={ns}; expected 2 each)")
        return mjcf

    def _action_dim(self) -> int:
        return 2                                                    # [throttle, steer]

    def _apply_control(self, a: np.ndarray) -> None:
        throttle, steer = float(a[0]), float(a[1])
        self.data.ctrl[self._rear_ctrl] = throttle * self.ctrl_range        # rear wheel ω target
        self.data.ctrl[self._steer_ctrl] = steer * self._max_steer          # front steer angle (rad)


# ── factories: the make_env callables the trainers/DAgger/eval consume ────────
def make_cheetah(*, max_steps: int = 300, ctrl_range: float = 80.0,
                 reward_spec: RewardSpec | None = None, **kw: Any) -> LeggedLocomotionEnv:
    """Planar half-cheetah runner. Planar base (slide-x/z + pitch); fall = torso below 0.20 m or tipped."""
    cfg = LocomotionConfig(plant="half_cheetah.hymeko", name="cheetah", root_body="torso",
                           base_mode="planar", ctrl_range=ctrl_range, frame_skip=5, max_steps=max_steps,
                           fall_height=0.20, flip_cos=-0.2, joint_damping=0.8, joint_armature=0.02,
                           reward_spec=reward_spec)
    return LeggedLocomotionEnv(cfg, gait_freq=2.0, gait_amp=0.8, **kw)


def make_humanoid(*, max_steps: int = 300, ctrl_range: float = 120.0,
                  reward_spec: RewardSpec | None = None, **kw: Any) -> LeggedLocomotionEnv:
    """Biped humanoid walk--run. Free base; fall = pelvis below 0.6 m or tipped past horizontal."""
    cfg = LocomotionConfig(plant="humanoid.hymeko", name="humanoid", root_body="pelvis",
                           base_mode="free", ctrl_range=ctrl_range, frame_skip=5, max_steps=max_steps,
                           fall_height=0.6, flip_cos=0.2, joint_damping=1.2, joint_armature=0.02,
                           reward_spec=reward_spec)
    return LeggedLocomotionEnv(cfg, gait_freq=1.3, gait_amp=0.5, **kw)


def make_vehicle(*, max_steps: int = 400, ctrl_range: float = 18.0, terrain: str | None = None,
                 terrain_relief: float = 0.12, **kw: Any) -> WheeledVehicleEnv:
    """Diff-drive vehicle on a high-speed waypoint track. Freejoint chassis; velocity-servo wheels
    (``ctrl_range`` = max wheel ω rad/s ≈ ctrl_range·wheel_radius m/s); real rollover terminates.
    ``terrain`` ∈ {None, "hills", "bumps", "ramps"} drives it over a procedural heightfield."""
    cfg = LocomotionConfig(plant="robot_4wh.hymeko", name="vehicle", root_body="base_link",
                           base_mode="free", ctrl_range=ctrl_range, frame_skip=5, max_steps=max_steps,
                           reach_radius=1.2, flip_cos=-0.3, joint_damping=0.05, joint_armature=0.02,
                           terrain=terrain, terrain_relief=terrain_relief, terrain_radius=14.0)
    return WheeledVehicleEnv(cfg, **kw)


# Effective driven-wheel radius of race_car.hymeko (cylinder radius 0.33 m) — ground speed ≈ wheel_ω · radius.
_RACE_WHEEL_RADIUS = 0.33


def make_race_car(*, top_speed_kmh: float = 220.0, max_steps: int = 1500, ramp_steps: int = 500,
                  course: str = "straight", circuit_scale: float = 60.0, **kw: Any) -> WheeledVehicleEnv:
    """Full-scale high-speed car (``race_car.hymeko``): long wheelbase, low CG, big wheels — stable at
    ~200 km/h (verified ~260 km/h upright). ``course``: ``"straight"`` (a drag strip) or ``"circuit"`` (a flat
    **Bézier** GP circuit — smooth curvature the diff-drive holds at speed, hitting ~200 km/h on the straights
    and easing through corners; a 55 m/s corner needs radius ≳ 300 m, hence the large ``circuit_scale``).
    ``ctrl_range`` is the wheel-ω cap from ``top_speed_kmh``; the expert ramps the throttle over ``ramp_steps``
    to avoid a standing-start wheelspin. Freejoint chassis; real rollover terminates."""
    ctrl_range = (top_speed_kmh / 3.6) / _RACE_WHEEL_RADIUS
    if course == "circuit":
        track = kw.pop("track", race_circuit(scale=circuit_scale))
        pursuit = DiffDrivePursuit(ramp_steps=ramp_steps, steer_gain=0.3, slowdown=6.0)
        reach = 16.0
    else:
        track = kw.pop("track", straight_track(60.0, 120.0, 30))
        pursuit = DiffDrivePursuit(ramp_steps=ramp_steps, steer_gain=0.4)
        reach = 3.0
    cfg = LocomotionConfig(plant="race_car.hymeko", name="racecar", root_body="base_link",
                           base_mode="free", ctrl_range=ctrl_range, frame_skip=5, max_steps=max_steps,
                           reach_radius=reach, flip_cos=-0.3, joint_damping=0.02, joint_armature=0.05)
    return WheeledVehicleEnv(cfg, track=track, pursuit=pursuit, **kw)


# Effective driven-wheel radius of f1tenth.hymeko (cylinder radius 0.05 m).
_F1TENTH_WHEEL_RADIUS = 0.05


def make_f1tenth(*, top_speed_ms: float = 6.0, max_steps: int = 500, max_steer: float = 0.4,
                 track: np.ndarray | None = None, terrain: str | None = None,
                 terrain_relief: float = 0.04, **kw: Any) -> AckermannCarEnv:
    """1/10-scale Ackermann race car (``f1tenth.hymeko``, the Tamiya F1/10 / F1TENTH platform). Realistic RC
    scale (wheelbase 0.32 m, wheel radius 0.05 m, ~3.4 kg, ~5–10 m/s — no 200 km/h). ``top_speed_ms`` sets the
    rear wheel-ω cap (ω = v / wheel_radius); ``max_steer`` (rad) the front steering limit. Default course is a
    small circular racing loop the Ackermann steering follows. ``terrain`` (small relief at 1/10 scale) drives
    it over a rally heightfield."""
    ctrl_range = top_speed_ms / _F1TENTH_WHEEL_RADIUS
    cfg = LocomotionConfig(plant="f1tenth.hymeko", name="f1tenth", root_body="base_link",
                           base_mode="free", ctrl_range=ctrl_range, frame_skip=5, max_steps=max_steps,
                           reach_radius=1.2, flip_cos=-0.3, joint_damping=0.0005, joint_armature=0.02,
                           terrain=terrain, terrain_relief=terrain_relief, terrain_radius=14.0)
    # A large gentle racing loop the Ackermann steering tracks cleanly (a tight loop makes it understeer/circle).
    course = track if track is not None else circular_track(0.0, 8.0, 8.0, 20)
    return AckermannCarEnv(cfg, track=course, max_steer=max_steer,
                           pursuit=AckermannPursuit(throttle=0.8, steer_gain=0.6, ramp_steps=80), **kw)


SUBSTRATES = {"cheetah": make_cheetah, "humanoid": make_humanoid,
              "vehicle": make_vehicle, "racecar": make_race_car, "f1tenth": make_f1tenth}
