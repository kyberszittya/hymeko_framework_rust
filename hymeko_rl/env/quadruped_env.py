"""Goal-reaching quadruped env on the HyMeKo-described four-legged robot (``data/robotics/quadruped.hymeko``).

Torso + four two-link legs (revolute hip + knee per leg, 8 actuated DOF). The objective is **reach a goal in
front of the robot as fast as possible, smoothly** — the reward pays for closing the distance and a success
bonus, and *penalises* time, joint velocity (thrashing), and joint-velocity change (jerk). The robot is
jump-capable, so a learned policy typically lunges/hops toward the goal.

The robot's attachment to the world is **declared in the ``.hymeko``** as the ``@base`` joint — this env does
not inject it. The env reads that joint *by name* and sets the base mode explicitly (``base=``):

  * ``"free"``  — promote the declared base joint to a MuJoCo ``<freejoint>`` (6-DOF floating base): the torso
                  is free to translate/rotate, so it can locomote toward the goal.
  * ``"fixed"`` — weld the torso to the world (a stationary test rig).

The reward reuses the declarative term registry (:mod:`hymeko_rl.env.reward`): ``reach_distance`` (−dist to
goal), ``success_bonus`` (reached), and the generic ``time_penalty`` / ``joint_velocity`` /
``joint_acceleration`` smoothness terms shared with the planar-grasp task. The env speaks the same gymnasium /
``(N, feat)`` per-vertex contract as the other HSiKAN testbeds.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import (
    actuated_dof_addrs,
    emit_arm_mjcf,
    strip_actuators,
    with_collision_floor,
)
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.hypergraph_state import HypergraphState

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_HYMEKO = _REPO / "data" / "robotics" / "quadruped.hymeko"

BaseMode = Literal["free", "fixed", "planar"]
TaskMode = Literal["goal", "stand"]

# Default "reach the goal fast and smooth" reward: dense −dist + a big sparse success bonus, with time
# pressure and the velocity/jerk smoothness penalties. Weights chosen so distance/success dominate and the
# smoothness terms refine (see reports/2026-06-22-fast-smooth-rewards.md).
GOAL_REWARD = RewardSpec((
    ("goal_progress", 20.0),        # +Δdist closed: the dense forward locomotion driver
    ("success_bonus", 50.0),        # +50 on the step the goal is reached (episode then ends)
    ("time_penalty", 0.1),          # −0.1 / step → minimise time, not just distance
    ("joint_velocity", 0.0005),     # −Σ q̇²  : gentle (don't suppress gait exploration)
    ("joint_acceleration", 0.002),  # −Σ(Δq̇)²: gentle smoothness / jerk
))

# "Stand at nominal height and don't fall" reward (task="stand"). Redesigned 2026-07-03 (Hajdu): the prior
# spec paid an UNCONDITIONAL `alive` bonus + upright, so a crouched/collapsed-but-not-inverted robot scored
# ~as high as a standing one (flip_cos=−0.2 rarely terminates), and height (`torso_height`) was dominated —
# falling was not penalised and Z was not the point, so stand_rate stayed ~0 while the policy "learned". The
# fix makes the reward's argmax the graded success predicate: the dominant `standing` term is +1 iff
# (upright > stand_cos AND |z − stand_height| < tol) — identical to the DwellMetric — so crouch/collapse/flop
# score 0, and `torso_height` (up-weighted) gives the dense pull toward standing height. reward ≡ metric.
STAND_REWARD = RewardSpec((
    ("standing", 5.0),           # +1 iff upright AND at nominal height — the EXACT success predicate (dominant)
    ("torso_height", 3.0),       # −|z − stand_height|: dense gradient to the standing height (Z is the point)
    ("upright", 1.0),            # +cos(tilt): dense gradient to a level torso
    ("stand_still", 0.1),        # −|v_x|: hold ground, don't drift/walk off
    ("joint_velocity", 0.001),   # −Σ q̇²: gentle, don't thrash the legs to balance
))


def set_base_mode(mjcf: str, joint_name: str, mode: BaseMode) -> str:
    """Set the robot's base attachment by rewriting the **declared** ``joint_name`` joint in ``mjcf``.

    The world-attachment is declared in the ``.hymeko`` (see ``quadruped.hymeko`` ``@base``); this promotes
    it rather than guessing a body name. Modes:

      * ``"free"``   — a ``<freejoint>`` (6-DOF floating base; can leave the ground / tip over sideways).
      * ``"planar"`` — slide-x + slide-z + pitch-y: the torso is **confined to the sagittal plane**, so it
                       cannot fall sideways — the easy-to-learn *walker* base (HalfCheetah-style).
      * ``"fixed"``  — remove the joint (weld the torso at its declared pose).

    All base DOF are passive (the legs do the work). # Preconditions ``mjcf`` contains exactly one
    ``<joint name="{joint_name}" .../>``. # Postconditions returns the rewritten MJCF; raises ``ValueError``
    if the declared joint is absent.
    """
    pattern = rf'<joint name="{re.escape(joint_name)}"[^>]*/>'
    if not re.search(pattern, mjcf):
        raise ValueError(f"declared base joint {joint_name!r} not found in MJCF "
                         f"(is it declared in the .hymeko?)")
    # The base is a passive anchor — never actuated. The emitter motors every non-fixed joint, so drop the
    # base motor first (else the replacement joints leave a dangling, non-actuatable transmission target).
    mjcf = strip_actuators(mjcf, [joint_name])
    if mode == "free":
        replacement = f'<freejoint name="{joint_name}"/>'
    elif mode == "planar":
        replacement = (f'<joint name="{joint_name}_x" type="slide" axis="1 0 0"/>'
                       f'<joint name="{joint_name}_z" type="slide" axis="0 0 1"/>'
                       f'<joint name="{joint_name}_pitch" type="hinge" axis="0 1 0"/>')
    else:  # fixed
        replacement = ""
    return re.sub(pattern, replacement, mjcf, count=1)


class QuadrupedGoalEnv(gym.Env[np.ndarray, np.ndarray]):
    """A HyMeKo quadruped that learns to reach a goal in front of it, fast and smoothly.

    # Preconditions ``frame_skip >= 1``, ``max_steps >= 1``, ``ctrl_range > 0``, ``goal_distance > 0``,
    ``reach_radius > 0``.
    # Invariants ``observation`` is ``(n_vertices, 2)`` float32: the torso vertex carries
    ``[signed_dx_to_goal, forward_velocity]``, each leg vertex its joint ``[qpos, qvel]``; ``action`` is the
    8 leg-motor commands normalised to ``[-1, 1]`` and scaled to ``±ctrl_range`` N·m internally.
    # Postconditions (``step``) ``terminated`` iff the goal is reached (``task="goal"`` and
    ``dist < reach_radius``) or the free-base torso tips past ``flip_cos``; ``reward`` is the configured
    :class:`RewardSpec` over ``dist`` and the action. ``info["standing"]`` is True iff the torso is upright
    (``> stand_cos``) at its nominal height (task-agnostic; the ``task="stand"`` success signal).

    # Tasks ``task="goal"`` reaches a point in front (default). ``task="stand"`` balances in place — use it
    with ``base="free"`` (the torso must be able to fall for standing to be a task) and it defaults to
    :data:`STAND_REWARD`.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, hymeko_path: str | Path = _DEFAULT_HYMEKO, base: BaseMode = "planar",
                 base_joint: str = "base", goal_distance: float = 1.5, reach_radius: float = 0.3,
                 ctrl_range: float = 50.0, frame_skip: int = 5, max_steps: int = 250,
                 init_noise: float = 0.02, leg_damping: float = 0.8, leg_armature: float = 0.02,
                 flip_cos: float = -0.2, task: TaskMode = "goal", stand_cos: float = 0.9,
                 stand_height_tol: float = 0.08, reward_spec: RewardSpec | None = None) -> None:
        super().__init__()
        if frame_skip < 1 or max_steps < 1 or ctrl_range <= 0 or goal_distance <= 0 or reach_radius <= 0:
            raise ValueError("frame_skip/max_steps>=1, ctrl_range/goal_distance/reach_radius>0 required")
        if stand_cos <= 0 or stand_height_tol <= 0:
            raise ValueError("stand_cos, stand_height_tol > 0 required")
        # Task selects the default reward: "goal" reaches a point in front (GOAL_REWARD); "stand" balances in
        # place (STAND_REWARD). A standing task is only meaningful with a base that CAN fall — use base="free".
        self.task: TaskMode = task
        self.stand_cos = float(stand_cos)
        self.stand_height_tol = float(stand_height_tol)
        default_reward = STAND_REWARD if task == "stand" else GOAL_REWARD
        self.reward_spec = reward_spec if reward_spec is not None else default_reward
        # One source -> scene + hypergraph. The base attachment is DECLARED in the .hymeko; we set its mode.
        mjcf = set_base_mode(emit_arm_mjcf(hymeko_path, name="quad"), base_joint, base)
        mjcf = self._tune_legs(mjcf, leg_damping, leg_armature)
        mjcf = self._seat_floor(mjcf)
        self._mjcf = mjcf                 # kept so the renderer can re-skin the scene (decorate_scene)
        self.base = base
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.hg = HypergraphState.from_mjcf(mjcf, is_path=False)
        self.torso = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"))
        if self.torso < 0:
            raise ValueError("expected a 'torso' body in the emitted quadruped")
        self.n_actions = int(self.model.nu)
        self._act_dofs = actuated_dof_addrs(self.model)        # leg joints (for smoothness terms)
        self._torso_vtx = self.torso - 1                       # torso's hypergraph vertex (body b -> b-1)
        # joint -> hypergraph vertex (child body b -> vertex b-1) + qpos/dof addresses.
        self._jnt_vtx = np.array([int(self.model.jnt_bodyid[j]) - 1
                                  for j in range(self.model.njnt)], dtype=np.int64)
        self._jnt_qadr = self.model.jnt_qposadr.astype(np.int64)
        self._jnt_dadr = self.model.jnt_dofadr.astype(np.int64)
        self.goal_distance = float(goal_distance)
        self.reach_radius = float(reach_radius)
        self.reach_thresh = float(reach_radius)                 # read by the success_bonus reward term
        self.ctrl_range = float(ctrl_range)
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.init_noise = float(init_noise)
        self.flip_cos = float(flip_cos)
        self._step = 0
        self._q0 = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        # Nominal standing height = the torso z at the rest pose. The stand reward/metric pay for holding it.
        self._stand_height = float(self.data.xpos[self.torso, 2])
        self._standing = False                                 # success predicate, refreshed each step() (stand task)
        self.goal = np.array([float(self.data.xpos[self.torso, 0]) + self.goal_distance, 0.0], np.float32)
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()
        self._prev_dist = self.dist_to_goal()
        self._prev_x = float(self.data.xpos[self.torso, 0])
        self._vx = 0.0                                          # forward speed, base-agnostic (Δx / dt)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.hg.n_vertices, 2), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actions,), np.float32)

    @staticmethod
    def _tune_legs(mjcf: str, damping: float, armature: float) -> str:
        """Add joint damping + armature to the leg hinges (simulation tuning the .hymeko geometry can't
        carry). Without damping the un-sprung legs fold under the torso instead of holding a stance."""
        return re.sub(
            r'(<joint name="(?:hip|knee)_[a-z]{2}" type="hinge" axis="[^"]*" range="[^"]*)"/>',
            rf'\1" damping="{damping}" armature="{armature}"/>', mjcf)

    @staticmethod
    def _seat_floor(mjcf: str) -> str:
        """Place a collision floor just under the lowest geom at the rest pose."""
        m = mujoco.MjModel.from_xml_string(mjcf)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        bottoms = [float(d.geom_xpos[g, 2] - m.geom_rbound[g]) for g in range(m.ngeom)
                   if int(m.geom_bodyid[g])]
        return with_collision_floor(mjcf, z=min(bottoms) - 0.01)

    def dist_to_goal(self) -> float:
        """Planar distance from the torso to the goal point."""
        return float(np.hypot(self.data.xpos[self.torso, 0] - self.goal[0],
                              self.data.xpos[self.torso, 1] - self.goal[1]))

    def node_features(self) -> np.ndarray:
        """Per-vertex ``(N, 2)`` obs. Each leg vertex carries ``[qpos, qvel]``; the torso vertex carries the
        **task-relevant** state, so the policy observes what it optimises:

          * ``task="goal"``  → ``[dx_to_goal, vx]`` (goal direction + forward speed).
          * ``task="stand"`` → ``[z - stand_height, upright]`` (height error + torso levelness).

        Base-agnostic (free / planar / fixed); passive base joints are skipped."""
        feat = np.zeros((self.hg.n_vertices, 2), dtype=np.float32)
        for j in range(self.model.njnt):
            v = int(self._jnt_vtx[j])
            if v == self._torso_vtx or not 0 <= v < self.hg.n_vertices:
                continue                                          # skip passive base joints on the torso
            qa, da = int(self._jnt_qadr[j]), int(self._jnt_dadr[j])
            feat[v, 0] = float(self.data.qpos[qa])                # leg joint: angle
            feat[v, 1] = float(self.data.qvel[da])                # leg joint: rate
        if self.task == "stand":
            feat[self._torso_vtx, 0] = float(self.data.xpos[self.torso, 2]) - self._stand_height
            feat[self._torso_vtx, 1] = self._torso_uprightness()
        else:
            feat[self._torso_vtx, 0] = self.goal[0] - float(self.data.xpos[self.torso, 0])
            feat[self._torso_vtx, 1] = self._vx
        return feat

    def _torso_uprightness(self) -> float:
        """Cosine between the torso's local +z and world +z (1 = level, 0 = on its side, <0 = inverted)."""
        return float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = self._q0
        self.data.qpos[:] += self.np_random.uniform(-self.init_noise, self.init_noise, self.model.nq)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.goal = np.array([float(self.data.xpos[self.torso, 0]) + self.goal_distance, 0.0], np.float32)
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()
        self._prev_dist = self.dist_to_goal()
        self._prev_x = float(self.data.xpos[self.torso, 0])
        self._vx = 0.0
        self._step = 0
        return self.node_features(), {"dist": self.dist_to_goal()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)   # normalised command in [-1, 1]
        self.data.ctrl[:] = a * self.ctrl_range                        # scaled to N·m
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        x = float(self.data.xpos[self.torso, 0])
        self._vx = (x - self._prev_x) / (self.frame_skip * float(self.model.opt.timestep))  # base-agnostic
        self._prev_x = x
        dist = self.dist_to_goal()
        # Pose predicates BEFORE the reward: the stand reward's `standing` term reads self._standing, so it must
        # reflect THIS step's post-physics pose (else it lags one step). Standing = torso level (upright) AND near
        # its nominal height — identical to the DwellMetric the stand task is graded on.
        upright = self._torso_uprightness()
        self._standing = bool(upright > self.stand_cos
                              and abs(float(self.data.xpos[self.torso, 2]) - self._stand_height)
                              < self.stand_height_tol)
        reward = self.reward_spec.evaluate(self, dist, a)              # goal_progress reads self._prev_dist
        self._prev_dist = dist                                        # advance for next step's progress
        self._prev_act_qvel = self.data.qvel[self._act_dofs].copy()    # for next step's jerk term
        # Goal is only "reached" in the goal task; the stand task has no goal point to arrive at (it balances
        # in place, so its only terminal is a fall).
        reached = bool(self.task == "goal" and dist < self.reach_radius)
        standing = self._standing
        # Flip-terminate only the free (jump/stand) base. The planar walker is HalfCheetah-style: NO termination
        # on tipping — it gets the whole horizon to make forward progress and can recover from a pitch.
        terminated = bool(reached or (self.base == "free" and upright < self.flip_cos))
        truncated = self._step >= self.max_steps
        info = {"dist": dist, "vx": self._vx, "reached": reached, "upright": upright, "x": x,
                "standing": standing}
        return self.node_features(), reward, terminated, truncated, info
