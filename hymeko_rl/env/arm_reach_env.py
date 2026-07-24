"""Reaching env: drive the arm's end-effector to a sampled target.

Observation = **per-vertex features on the kinematic hypergraph** ``(N, feat)``: each
link node carries its driving joint's ``(qpos, qvel)`` plus the (broadcast) target
end-effector position. Both policies read this same obs — the HSiKAN backbone message-
passes over the hypergraph, the MLP flattens it — so the architecture comparison is a
pure backbone swap.

Scene source: two interchangeable arms behind one env. The hand-authored ``arm_world``
arm (4 DOF, mixed Z/Y/Y/Z, control-mode ablations) is the default; the **canonical** arm
is emitted from a ``.hymeko`` robot via :meth:`ArmReachEnv.from_hymeko` — the same source
that yields the kinematic hypergraph and the observation space (B-004/B-005 fixed
2026-06-19, so the emitted arm is now articulated at its native 6 DOF). ``HypergraphState
.from_mjcf`` derives the hypergraph from whichever MJCF is used; the emitted arm leaves
joint/actuator bounds unspecified, so the env supplies torque/joint-range fallbacks.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import (
    CONTROL_MODES, emit_arm_mjcf, make_arm_mjcf, slim_arm_collision, with_collision_floor,
)
from hymeko_rl.env.observation import REACH_OBSERVATION, ObservationSpec
from hymeko_rl.env.reward import REACH_REWARD, RewardSpec
from hymeko_rl.env.safety import CLEAN_SAFETY, SafetyState, compute_safety
from hymeko_rl.env.termination import DEATH_ON_CONTACT, TerminationSpec
from hymeko_rl.agents.hypergraph_state import HypergraphState

# The default reaching obs width (per vertex): [qpos, qvel, target(3), ee_error(3)] = 8.
# Derived from the spec so the two never drift; kept as a module constant because
# bc.py / tests size the default policy against it.
_NODE_FEAT = REACH_OBSERVATION.obs_dim

# Defaults the env supplies when a scene leaves bounds unspecified — the canonical
# hymeko-emitted arm declares neither joint ranges nor actuator ctrlranges (the
# emitter does not yet propagate the `limit -> joint_rev_limit` ref). The hand-authored
# arm_world arm specifies both, so for it these fallbacks are inert.
_DEFAULT_CTRL = 25.0      # torque bound (N·m) — matches arm_world's torque ctrlrange
_DEFAULT_QRANGE = 1.5     # rad — joint sampling bound for target/start configs


def _arm_home_floor_z(mjcf: str, *, margin: float = 0.02) -> float:
    """The z at which to seat a ground plane: just below the arm's lowest geom extent at the home
    pose (``qpos = 0``), so resting is contact-free and driving links lower is a real ground
    contact. Uses each geom's bounding radius (conservative — always at or below the true bottom).

    # Preconditions ``mjcf`` is a loadable scene (no floor required).
    # Postconditions a height ``<= 0`` for a mount-at-origin arm; ``0`` if the scene has no geoms.
    """
    m = mujoco.MjModel.from_xml_string(mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bottoms = [float(d.geom_xpos[g, 2] - m.geom_rbound[g])
               for g in range(m.ngeom) if int(m.geom_bodyid[g]) != 0]
    return (min(bottoms) - margin) if bottoms else 0.0


def _bounds_with_fallback(
    limited: np.ndarray, lo: np.ndarray, hi: np.ndarray, default: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element ``[lo, hi]`` where the element is genuinely limited, else
    ``[-default, +default]``. Used for both joint ranges and actuator ctrlranges.

    # Preconditions ``limited``, ``lo``, ``hi`` are same-length 1-D arrays.
    # Postconditions Returns float32 ``(lo, hi)`` with ``hi > lo`` elementwise.
    """
    lo = np.asarray(lo, dtype=np.float32).copy()
    hi = np.asarray(hi, dtype=np.float32).copy()
    degenerate = (~np.asarray(limited, dtype=bool)) | (hi <= lo)
    lo[degenerate] = -float(default)
    hi[degenerate] = float(default)
    return lo, hi


class ArmReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """End-effector reaching on the 4-DOF articulated arm.

    # Preconditions ``frame_skip >= 1``, ``max_steps >= 1``, ``reach_thresh > 0``.
    # Invariants ``observation`` is ``(n_vertices, 5)`` float32; ``action`` is
    ``(n_actions,)`` joint targets clipped to the actuator ``ctrlrange``. The target is a
    forward-kinematics-reachable EE position (sampled from a random joint config).
    """

    metadata = {"render_modes": []}

    def __init__(self, *, control_mode: str = "torque", mjcf: str | None = None,
                 obs_spec: ObservationSpec | None = None,
                 reward_spec: RewardSpec | None = None, frame_skip: int = 5,
                 max_steps: int = 80, reach_thresh: float = 0.06,
                 expert_gain: float = 1.0, ee_body: str = "flange_link",
                 reach_min_radius: float = 0.0, enable_safety: bool = False,
                 termination_spec: TerminationSpec | None = None) -> None:
        super().__init__()
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be in {CONTROL_MODES}; got {control_mode!r}")
        # Observation + reward = declarative specs (defaults: the reaching layout / dense
        # -distance). node_features() assembles the obs; step() evaluates the reward.
        self.obs_spec = obs_spec if obs_spec is not None else REACH_OBSERVATION
        self.reward_spec = reward_spec if reward_spec is not None else REACH_REWARD
        if frame_skip < 1 or max_steps < 1 or reach_thresh <= 0:
            raise ValueError("frame_skip/max_steps >= 1 and reach_thresh > 0 required")
        # Scene source (Strategy): an explicit `mjcf` (e.g. a hymeko-emitted arm) or
        # the hand-authored arm_world arm for `control_mode`. The caller is responsible
        # for `mjcf` matching `control_mode`'s actuator type (the emitted arm is torque).
        if mjcf is None:
            mjcf = make_arm_mjcf(control_mode)
        # Safety is opt-in: only then inject a ground plane / detect death / penalise config.
        # Off by default so the plain reach env (BC comparison, emitted-arm tests) is unchanged.
        self.enable_safety = bool(enable_safety)
        if self.enable_safety:
            mjcf = slim_arm_collision(mjcf)     # kill spurious non-adjacent self-collision
            # Seat the ground plane just below the arm's home extent (the emitted lower link clips
            # z=0 at its mount). Plane height must be set at compile time — runtime geom_pos does
            # not move a plane's collision surface. Skipped if the scene already has a floor.
            floor_z = 0.0 if 'type="plane"' in mjcf else _arm_home_floor_z(mjcf)
            mjcf = with_collision_floor(mjcf, z=floor_z)
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.hg = HypergraphState.from_mjcf(mjcf, is_path=False)
        # the ground plane geom (for ground-contact detection); -1 if somehow absent.
        self._floor_geom = next(
            (g for g in range(self.model.ngeom)
             if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), -1)
        self.control_mode = control_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.reach_thresh = reach_thresh
        self.expert_gain = expert_gain
        # computed-torque expert: acceleration-level PD gains (torque mode, scaled by the
        # mass matrix); velocity-command gain (velocity mode).
        self._exp_kp, self._exp_kv, self._exp_vgain = 3000.0, 110.0, 8.0
        self.n_actions = int(self.model.nu)
        self._ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self._ee < 0:
            raise ValueError(f"end-effector body {ee_body!r} not in the model")
        # joint -> hypergraph vertex (its child body, remapped: body b -> vertex b-1).
        self._jnt_vtx = np.array(
            [int(self.model.jnt_bodyid[j]) - 1 for j in range(self.model.njnt)],
            dtype=np.int64)
        # Control + joint-sampling bounds, with a fallback for scenes that leave them
        # unspecified (the hymeko-emitted arm); inert for arm_world, which sets both.
        self._ctrl_lo, self._ctrl_hi = _bounds_with_fallback(
            self.model.actuator_ctrllimited,
            self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1],
            _DEFAULT_CTRL)
        self._q_lo, self._q_hi = _bounds_with_fallback(
            self.model.jnt_limited,
            self.model.jnt_range[:, 0], self.model.jnt_range[:, 1], _DEFAULT_QRANGE)
        self._target = np.zeros(3, dtype=np.float32)
        self._step = 0
        self.reach_min_radius = float(reach_min_radius)
        # The death predicate (from the model when provided, else ground-contact ∨ self-collision).
        self.termination_spec = termination_spec if termination_spec is not None else DEATH_ON_CONTACT
        self._last_safety: SafetyState = CLEAN_SAFETY
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.hg.n_vertices, self.obs_spec.obs_dim),
            dtype=np.float32)
        # action = raw control input for the chosen actuator (torque / position / velocity),
        # bounded by the actuator's ctrlrange.
        self.action_space = spaces.Box(self._ctrl_lo, self._ctrl_hi, dtype=np.float32)

    @classmethod
    def from_hymeko(cls, hymeko_path: str, *, name: str = "arm",
                    control_mode: str = "torque", obs_profile: str | None = None,
                    task_profile: str | None = None, ee_body: str = "tool",
                    **kwargs: Any) -> "ArmReachEnv":
        """Build the reaching env from a ``.hymeko`` robot — the canonical scene path.

        The MJCF is emitted via the CLI (``emit_arm_mjcf``), so the *same* source produces
        the MuJoCo scene, the kinematic hypergraph, and — when ``obs_profile`` /
        ``task_profile`` are given — the observation layout and the reward, all from
        ``.hymeko``. ``control_mode`` selects the actuator interface (torque ``<motor>`` as
        emitted, or position / velocity servos retargeted on the canonical geometry).

        # Preconditions ``hymeko_path`` (and any profile given) exist; the CLI is built.
        # Postconditions Returns an env whose model is the emitted robot, whose observation
        follows ``obs_profile`` and whose reward follows ``task_profile`` (defaults: the
        reaching layout / dense -distance).
        """
        mjcf = emit_arm_mjcf(hymeko_path, name=name, control_mode=control_mode)
        obs_spec = ObservationSpec.from_hymeko(obs_profile) if obs_profile else None
        reward_spec = RewardSpec.from_hymeko(task_profile) if task_profile else None
        # Optional: the death predicate from the task profile (absent => env default).
        term_spec: TerminationSpec | None = None
        if task_profile:
            try:
                term_spec = TerminationSpec.from_hymeko(task_profile)
            except ValueError:
                term_spec = None   # profile declares no termination_spec
        return cls(mjcf=mjcf, control_mode=control_mode, ee_body=ee_body,
                   obs_spec=obs_spec, reward_spec=reward_spec, termination_spec=term_spec, **kwargs)

    # -- helpers ----------------------------------------------------------------
    def _ee_pos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._ee], dtype=np.float32)

    def _fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: EE position at joint config ``q`` (no side effects)."""
        saved = self.data.qpos.copy()
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        ee = self._ee_pos()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return ee

    def node_features(self) -> np.ndarray:
        """Current obs: per-vertex ``(N, obs_dim)`` features on the kinematic hypergraph,
        assembled from the declared :class:`ObservationSpec` channels (default: each node's
        joint ``(qpos, qvel)``, the broadcast target EE position, and the **live EE error**
        ``target − ee``). The error is what the closed-loop expert acts on directly; without
        it the policy must learn forward kinematics to recover it. The layout is the spec's
        — no hand-coded column offsets."""
        return self.obs_spec.assemble(self)

    # -- gymnasium API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        lo, hi = self._q_lo, self._q_hi
        info = self._reset_target(lo, hi)                    # overridable: a reachable EE position (base) or pose (SE3 subclass)
        self.data.qpos[:] = self._start_config(lo, hi)       # overridable start (base: near-home)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._last_safety = CLEAN_SAFETY
        return self.node_features(), info

    def _reset_target(self, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
        """Sample the task target and return the reset ``info`` dict. Base = a forward-kinematics-reachable EE
        POSITION; the SE(3) subclass overrides to a full reachable pose. # Postconditions sets ``self._target``;
        returns ``{'target': ...}`` (RNG-order identical to the prior inline body)."""
        self._target = self._sample_target(lo, hi).copy()   # reachable AND outside the robot
        return {"target": self._target.copy()}

    def _start_config(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Initial joint configuration for an episode. Base = near-home (``uniform·0.3``); the SE(3) subclass couples
        the start to the sampled target config so the 6-D pose error is closable. # Postconditions length ``nq``."""
        return self.np_random.uniform(lo, hi) * 0.3

    def _sample_target(self, lo: np.ndarray, hi: np.ndarray, *, attempts: int = 64) -> np.ndarray:
        """A forward-kinematics-reachable EE target whose horizontal distance from the base axis is
        ``>= reach_min_radius`` — i.e. outside the robot's vertical column. Capped reject sampling;
        falls back to the farthest candidate seen if none clears the radius (no infinite loop).

        # Postconditions returns a length-3 world position; with ``reach_min_radius == 0`` the first
        sample is accepted (RNG-identical to the prior single-draw behaviour).
        """
        best = self._fk(self.np_random.uniform(lo, hi).astype(np.float32))
        best_r = float(np.hypot(best[0], best[1]))
        if best_r >= self.reach_min_radius:
            return best
        for _ in range(attempts - 1):
            cand = self._fk(self.np_random.uniform(lo, hi).astype(np.float32))
            r = float(np.hypot(cand[0], cand[1]))
            if r >= self.reach_min_radius:
                return cand
            if r > best_r:
                best_r, best = r, cand
        return best   # farthest seen; reach_min_radius unreachable in this workspace

    @property
    def expert_action(self) -> np.ndarray:
        """Closed-loop demonstrator, in the control space of ``control_mode``.

        A damped-least-squares IK step gives a desired joint config ``q_des = q + Δq``
        (``Δq = gain·Jₚᵀ(Jₚ Jₚᵀ + λ²I)⁻¹(target − ee)``, a well-posed single-valued
        function of state). That target is then realised in the chosen control space:
        **torque** — computed torque ``kp(q_des−q) − kv·q̇ + qfrc_bias`` (gravity/Coriolis
        compensated); **position** — the target ``q_des``; **velocity** — ``vgain·Δq``.
        """
        n = self.n_actions
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self._ee_pos(), self._ee)
        jp = jacp[:, :n]
        err = self._target - self._ee_pos()
        dq = self.expert_gain * jp.T @ np.linalg.solve(jp @ jp.T + 0.05 ** 2 * np.eye(3), err)
        q, qd = self.data.qpos[:n], self.data.qvel[:n]
        q_des = q + dq
        if self.control_mode == "torque":
            # inverse dynamics: τ = M(q)·a_des + bias (bias = Coriolis + gravity).
            a_des = (self._exp_kp * (q_des - q) - self._exp_kv * qd).astype(np.float64)
            ma = np.zeros(self.model.nv)
            mujoco.mj_mulM(self.model, self.data, ma, a_des)
            a = ma[:n] + self.data.qfrc_bias[:n]
        elif self.control_mode == "velocity":
            a = self._exp_vgain * dq
        else:  # position
            a = q_des
        return np.asarray(np.clip(a, self._ctrl_lo, self._ctrl_hi), dtype=np.float32)

    def step(self, action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # action is the raw actuator control (torque/position/velocity), clipped to range.
        ctrl = np.clip(np.asarray(action, dtype=np.float32), self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        dist = float(np.linalg.norm(self._ee_pos() - self._target))
        # Death conditions (terminate): ground contact or self-collision — only when safety is on.
        if self.enable_safety:
            self._last_safety = self.safety_state()
            death = self.termination_spec.is_terminal(self._last_safety)
        else:
            death = False
        terminated = bool(self._reached(dist) or death)
        truncated = self._step >= self.max_steps
        reward = self.reward_spec.evaluate(self, dist, ctrl) + self._extra_reward(dist, ctrl)
        info = {"dist": dist, "ee": self._ee_pos().copy(), "safety": self._last_safety, "death": death}
        info.update(self._extra_step_info())
        return (self.node_features(), reward, terminated, truncated, info)

    # -- Template-Method hooks (base = position reach; SE3ReachEnv overrides for pose) ---------
    def _reached(self, dist: float) -> bool:
        """Task-success predicate on the current state. Base = EE within ``reach_thresh``; the SE(3) subclass adds the
        orientation gate. # Postconditions pure (no side effects)."""
        return bool(dist < self.reach_thresh)

    def _extra_reward(self, dist: float, ctrl: np.ndarray) -> float:
        """Additional reward beyond ``reward_spec`` (base: none; SE3: a weighted orientation-error penalty)."""
        return 0.0

    def _extra_step_info(self) -> dict[str, Any]:
        """Extra ``info`` keys for ``step`` (base: none; SE3: the orientation error)."""
        return {}

    def safety_state(self) -> SafetyState:
        """Current ground-contact / self-collision / joint-margin / below-ground state, derived
        from the live MuJoCo contacts and ``qpos`` (see :func:`hymeko_rl.env.safety.compute_safety`)."""
        return compute_safety(self.model, self.data, self._floor_geom)

