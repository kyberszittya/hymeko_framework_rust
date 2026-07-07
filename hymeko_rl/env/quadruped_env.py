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
    emit_arm_mjcf,
    strip_actuators,
    with_collision_floor,
)
from hymeko_rl.env._profile import read_scene_fields
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.agents.hypergraph_state import HypergraphState

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_HYMEKO = _REPO / "data" / "robotics" / "quadruped.hymeko"
_DEFAULT_STAND_SCENARIO = _REPO / "data" / "robotics" / "quadruped_stand.hymeko"

BaseMode = Literal["free", "fixed", "planar"]
TaskMode = Literal["goal", "stand"]

# The RL-controlled leg joints (hip_abduct/hip_flex/knee × 4). Every OTHER emitted joint on the Aibo plant is
# an ERS-1000 expressive DOF (head 3 / neck / waist / mouth / ears 2 / tail 2), held at neutral by a position
# servo (see QuadrupedGoalEnv._hold_expressive_joints) so the action space is the 12 legs. Name PREFIXES: the
# per-leg suffix (fl/fr/bl/br) varies.
_LEG_JOINT_PREFIXES = ("hip_abduct", "hip_flex", "knee")

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
                 stand_height_tol: float = 0.08, pd_kp: float = 40.0, pd_kd: float = 2.0,
                 reward_spec: RewardSpec | None = None) -> None:
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
        mjcf = self._hold_expressive_joints(mjcf)   # 10 expressive DOF held at neutral; RL drives the 12 legs
        self._mjcf = mjcf                 # kept so the renderer can re-skin the scene (decorate_scene)
        self.base = base
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.hg = HypergraphState.from_mjcf(mjcf, is_path=False)
        self.torso = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"))
        if self.torso < 0:
            raise ValueError("expected a 'torso' body in the emitted quadruped")
        # The RL action space is the CONTROLLED (leg) actuators only; the expressive actuators are position
        # servos holding neutral (see _hold_expressive_joints). `_leg_ctrl_idx` maps the action to those motors.
        self._leg_ctrl_idx = self._controlled_actuators()
        self.n_actions = int(self._leg_ctrl_idx.size)
        if self.n_actions == 0:
            raise ValueError("no leg actuators found — is the quadruped plant emitted with leg joints?")
        self._act_dofs = np.array(                             # leg dofs (for smoothness terms + obs)
            [int(self.model.jnt_dofadr[int(self.model.actuator_trnid[i, 0])]) for i in self._leg_ctrl_idx],
            dtype=np.int64)
        self._leg_qadr = np.array(                              # leg qpos addresses (for the PD-hold expert)
            [int(self.model.jnt_qposadr[int(self.model.actuator_trnid[i, 0])]) for i in self._leg_ctrl_idx],
            dtype=np.int64)
        self.pd_kp, self.pd_kd = float(pd_kp), float(pd_kd)
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

    @classmethod
    def from_hymeko(cls, scenario: str | Path = _DEFAULT_STAND_SCENARIO, *,
                    hymeko_path: str | Path = _DEFAULT_HYMEKO) -> "QuadrupedGoalEnv":
        """Build the standing MDP END-TO-END from a ``.hymeko`` scenario — the declarative-substrate entry.

        The ``scenario`` file declares the balance-env tuning (its ``scene`` scalars) and the objective (its
        ``reward_spec``, read by :meth:`RewardSpec.from_hymeko`); the plant is the imported Aibo robot
        (``hymeko_path``, emitted separately). ``base="free"`` / ``task="stand"`` are structural to a standing
        task (a fixed base cannot fall) and are fixed here. The training strategy declared in the *same* file is
        read separately by :meth:`hymeko_rl.experiments.training_spec.TrainingSpec.from_hymeko` (the trainer, not
        the env), so one ``.hymeko`` carries plant + tuning + objective + strategy.

        # Preconditions ``scenario`` declares one ``scene`` bundle and one ``reward_spec`` bundle.
        # Postconditions returns a stand-mode env whose reward and tuning come entirely from ``scenario``.
        # Errors ``FileNotFoundError``; ``ValueError`` (missing/multiple ``scene``/``reward_spec``, bad field).
        """
        f = read_scene_fields(scenario, "scene")

        def scalar(name: str, default: float) -> float:
            """A scene scalar; a vector-valued field here is a declaration error, not a silent coercion."""
            v = f.get(name, default)
            if isinstance(v, tuple):
                raise ValueError(f"{scenario}: scene field {name!r} must be a scalar, got a vector {v}")
            return float(v)

        return cls(
            hymeko_path=hymeko_path, base="free", task="stand",
            stand_cos=scalar("stand_cos", 0.9),
            stand_height_tol=scalar("stand_height_tol", 0.08),
            ctrl_range=scalar("ctrl_range", 50.0),
            frame_skip=int(scalar("frame_skip", 5)),
            max_steps=int(scalar("max_steps", 250)),
            flip_cos=scalar("flip_cos", -0.2),
            leg_damping=scalar("leg_damping", 0.8),
            leg_armature=scalar("leg_armature", 0.02),
            init_noise=scalar("init_noise", 0.02),
            reward_spec=RewardSpec.from_hymeko(scenario),
        )

    @staticmethod
    def _tune_legs(mjcf: str, damping: float, armature: float) -> str:
        """Add joint damping + armature to the leg hinges (simulation tuning the .hymeko geometry can't
        carry). Without damping the un-sprung legs fold under the torso instead of holding a stance.

        Matches the Aibo dog's three leg-joint families (``hip_abduct`` / ``hip_flex`` / ``knee``), each
        suffixed by a two-letter leg id (``fl``/``fr``/``bl``/``br``); the passive ``base`` carrier is left
        untouched (it is promoted/stripped by :func:`set_base_mode`)."""
        return re.sub(
            r'(<joint name="(?:hip_abduct|hip_flex|knee)_[a-z]{2}" type="hinge" axis="[^"]*" range="[^"]*)"/>',
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

    @staticmethod
    def _hold_expressive_joints(mjcf: str, stiffness: float = 8.0, damping: float = 0.8,
                                armature: float = 0.02) -> str:
        """Hold every non-leg joint at its neutral pose *passively* (spring-to-neutral) instead of RL-driving it.

        The emitter motors every non-fixed joint, but the ERS-1000's 10 expressive DOF (head/neck/waist/mouth/
        ears/tail) must not be RL-driven at leg-scale torque — a 50 N·m command on a ~10 g ear yields absurd
        accelerations that blow up the reward. Rather than a stiff position servo (numerically unstable on the
        near-massless head-gimbal links — QACC blow-up), each expressive joint is made **passive**: its torque
        motor is stripped and the joint element gets a ``stiffness`` spring to neutral (``springref`` 0), plus
        ``damping`` and — critically — ``armature`` (reflected rotor inertia) so the stiff hold on a light link
        stays stable. This keeps the Aibo posture (head up, ears up, tail up) and confines the action space to the
        12 legs. The expressive set is derived from the emitted joints (any non-leg ``<joint>``), so it needs no
        hardcoded list. # Postconditions non-leg motors removed; non-leg joints spring-held; leg joints untouched.
        """
        expressive = [j for j in re.findall(r'<joint name="([^"]+)"', mjcf)
                      if not j.startswith(_LEG_JOINT_PREFIXES)]
        if not expressive:
            return mjcf
        mjcf = strip_actuators(mjcf, expressive)                  # make the expressive joints passive
        held = set(expressive)
        hold = f' stiffness="{stiffness}" damping="{damping}" armature="{armature}"/>'

        def spring(m: "re.Match[str]") -> str:
            return m.group(0)[:-2] + hold if m.group("name") in held else m.group(0)

        return re.sub(r'<joint name="(?P<name>[^"]+)"[^>]*/>', spring, mjcf)

    def _controlled_actuators(self) -> np.ndarray:
        """The ``ctrl`` indices of the leg (torque) actuators — the RL action space (the expressive joints are
        passive springs, not actuators). # Postconditions int64 array of the leg-motor ctrl indices.
        """
        idx = []
        for i in range(int(self.model.nu)):
            jid = int(self.model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if jname.startswith(_LEG_JOINT_PREFIXES):
                idx.append(i)
        return np.asarray(idx, dtype=np.int64)

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

    @property
    def expert_action(self) -> np.ndarray:
        """A scripted **PD-hold-q0 standing demonstrator**: leg torques that hold each leg joint at its rest
        angle (the standing stance), normalised to the ``[-1, 1]`` action space. This is the standing-task BC
        demo source — the motivated lever after pure TD3 was *measured* to diverge on this task
        (``reports/2026-07-03-quadruped-gpu-cudagraph-fix.md``). Only the 12 legs are commanded; the expressive
        joints are spring-held by the plant.

        ``tau = clip(-kp·(q_leg − q0_leg) − kd·q̇_leg) / ctrl_range``.
        # Postconditions returns shape ``(n_actions,)`` in ``[-1, 1]``.
        """
        q = self.data.qpos[self._leg_qadr]
        qd = self.data.qvel[self._act_dofs]
        tau = -self.pd_kp * (q - self._q0[self._leg_qadr]) - self.pd_kd * qd
        return np.asarray(np.clip(tau / self.ctrl_range, -1.0, 1.0), dtype=np.float32)

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
        self.data.ctrl[self._leg_ctrl_idx] = a * self.ctrl_range       # torque to the 12 legs; expressive held

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
