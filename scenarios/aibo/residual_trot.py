"""Bounded residual over the trot-gait scaffold — learn to reach MULTIPLE goal positions.

The clock-driven ``SteeredTrotGait`` walks the AIBO but reaches only straight, near goals (measured
~1/6 across a distance×bearing grid: it steers poorly off-axis and closes distance slowly). This env
makes ``a = 0`` the pure scaffold (gait + heading pursuit, under the motion contract) and trains a
**bounded residual** (coin-R8 regime: ``final = clip(base + scale·a, ±1)``) to improve goal-reaching
across a *distribution* of goal positions — evaluated on **held-out** positions.

Reward = distance progress + heading alignment + a reach bonus − control cost; a fall terminates. The
residual is bounded so the safe scaffold (never falls, upright ≈ 1.0) is preserved — the coin-R8
prerequisite. Gym-like interface (``reset``/``step``, ``observation_space``/``action_space``) so the
repo SAC (`build_sac`/`train_sac`) drives it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .locomotion_gait import GAIT_PHASES, SteeredTrotGait, heading_error
from .motion_contract import JointVelocityGovernor

LEGS = ("fl", "fr", "bl", "br")  # leg order (matches _DIAG_PHASE and the hip_abduct_{leg} joints)
_LEFT = (0, 2)                    # fl, bl are the LEFT legs (fr, br are right) — the crab symmetry axis


def minimal_leg_hypergraph(symmetric: bool = False):
    """A 5-vertex leg hypergraph (torso + 4 legs) — the crab-relevant structure only, ~6× cheaper than
    the full 33-vertex body hg. ``symmetric=False``: plain kinematic arcs torso↔leg (down +1 / up −1).
    ``symmetric=True``: the torso↔leg SIGNS encode the LEFT/RIGHT symmetry axis (left legs fl,bl +1;
    right legs fr,br −1), so the signed propagation routes the torso's lateral goal-demand
    DIFFERENTIALLY to the two sides — a symmetric crab the flat MLP cannot represent."""
    from hymeko_rl.agents.hypergraph_state import HypergraphState
    labels = ("torso", "fl", "fr", "bl", "br")
    edges = np.array([(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0)], dtype=np.int64)
    if symmetric:                                        # left legs (1,3) +1, right legs (2,4) −1
        signs = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.int64)
        tag = "aibo_leg_min_sym_v1"
    else:
        signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)
        tag = "aibo_leg_min_v1"
    return HypergraphState(labels, edges, signs, topo_hash=tag)


@dataclass(frozen=True)
class ResidualTrotConfig:
    """Task distribution + reward shaping for the residual-over-trot goal-reaching problem."""

    dist_lo: float = 0.5
    dist_hi: float = 0.75
    bearing_deg: float = 40.0            # goals sampled in bearing ∈ [−bearing_deg, +bearing_deg]
    residual_mode: str = "leg"           # "leg" = 12-dim raw-target residual | "steer" = 2-dim (Δyaw, Δdrive) gait-param residual | "phase" = 12-dim residual PHASE-GATED per leg | "omni" = 4-dim per-leg ABDUCTION amplitude (phase-locked lateral crab over the forward trot — the RICHER action space, adds lateral DOF the trot leaves unused)
    abd_scale: float = 0.5               # omni mode: bound on the learned per-leg abduction (lateral) amplitude
    obs_mode: str = "flat"               # "flat" = 9-D vector (MLP) | "hypergraph" = (n_vertices, 4) per-vertex on the body's kinematic hypergraph (for signedkan/hsikan structure propagation)
    leg_hg_symmetric: bool = False       # leg_hypergraph mode: encode the LEFT/RIGHT symmetry axis in the hg signs
    gait_phase: str = "diag"             # base-gait phase pattern (GAIT_PHASES): "diag" (trot, asymmetric — default) | "bound" (front/back, instantaneously LEFT-RIGHT SYMMETRIC — the symmetric-scaffold test) | "pace" | "pronk"
    mirror_augment: bool = False         # omni/flat: randomly present the LEFT-RIGHT-MIRRORED task each episode → a symmetry-preserved policy that reaches BOTH crab sides (breaks the symmetry-breaking one-sided optimum)
    residual_scale: float = 0.25         # bounded residual (coin-R8): a small correction over the gait
    yaw_res_scale: float = 0.5           # steer mode: bound on the learned steering correction (rad)
    drive_res_scale: float = 0.5         # steer mode: bound on the learned speed correction
    reach_radius: float = 0.12
    max_steps: int = 800
    v_max: float = 8.0                   # motion-contract joint-speed cap
    progress_w: float = 12.0
    heading_w: float = 0.5
    reach_bonus: float = 8.0
    ctrl_w: float = 0.004
    fall_upright: float = 0.4


class _Box:
    """Minimal Box space (shape + uniform sample) — enough for the repo SAC driver."""

    def __init__(self, dim, low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
        self.shape = dim if isinstance(dim, tuple) else (dim,)
        self._lo, self._hi = low, high
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self._lo, self._hi, size=self.shape).astype(np.float32)


@dataclass
class ResidualTrotEnv:
    """Trot-scaffold + bounded residual, goal-reaching over a distribution of goal positions.

    # Preconditions
    ``a`` (action) is a 12-vector in [−1, 1] (a bounded residual over the gait). # Invariants: the
    scaffold action (``a = 0``) is applied under the motion contract and never overridden by more than
    ``residual_scale``; a fall terminates the episode.
    """

    cfg: ResidualTrotConfig = field(default_factory=ResidualTrotConfig)
    seed: int = 0
    _env: QuadrupedGoalEnv = field(init=False, repr=False)
    _gait: SteeredTrotGait = field(init=False, repr=False)
    _gov: JointVelocityGovernor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=self.cfg.dist_hi,
                                     reach_radius=self.cfg.reach_radius, max_steps=self.cfg.max_steps)
        if self.cfg.gait_phase not in GAIT_PHASES:
            raise ValueError(f"gait_phase must be one of {tuple(GAIT_PHASES)}; got {self.cfg.gait_phase!r}")
        self._phase_pat = GAIT_PHASES[self.cfg.gait_phase]        # per-leg gait phase (diag=asymmetric, bound=symmetric)
        self._gait = SteeredTrotGait(phase=self._phase_pat)
        self._gov = JointVelocityGovernor(v_max=self.cfg.v_max)
        self._rng = np.random.default_rng(self.seed)
        self._prev_dist = 0.0
        self._step_i = 0
        _dims = {"leg": 12, "phase": 12, "steer": 2, "omni": 4}
        act_dim = _dims[self.cfg.residual_mode]
        self.action_space = _Box(act_dim, seed=self.seed)
        if self.cfg.obs_mode == "hypergraph":
            self._abd_vtx = self._abduction_vertices()
            self.hg = self._env.hg                                # the body's full kinematic hypergraph
            self._n_vtx = int(self._env.hg.n_vertices)
            self.observation_space = _Box((self._n_vtx, 4), low=-5.0, high=5.0)  # (N vertices, feat)
        elif self.cfg.obs_mode == "leg_hypergraph":
            self.hg = minimal_leg_hypergraph(symmetric=self.cfg.leg_hg_symmetric)  # 5-vertex leg-only hg
            self._abd_vtx = [1, 2, 3, 4]                          # the 4 leg vertices (torso is vertex 0)
            self._n_vtx = 5
            self._abd_dof = [int(self._env.model.jnt_dofadr[mujoco.mj_name2id(
                self._env.model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_abduct_{leg}")]) for leg in LEGS]
            self.observation_space = _Box((5, 4), low=-5.0, high=5.0)
        else:
            self.observation_space = _Box(9, low=-5.0, high=5.0)
        self.max_steps = self.cfg.max_steps
        self.model = self._env.model

    # -- goal sampling + observation ------------------------------------------
    def _sample_goal(self) -> None:
        d = float(self._rng.uniform(self.cfg.dist_lo, self.cfg.dist_hi))
        b = float(self._rng.uniform(-self.cfg.bearing_deg, self.cfg.bearing_deg)) * np.pi / 180.0
        tx = float(self._env.data.xpos[self._env.torso, 0])
        ty = float(self._env.data.xpos[self._env.torso, 1])
        self._env.goal = np.array([tx + d * np.cos(b), ty + d * np.sin(b)], np.float32)
        self._env._prev_dist = self._env.dist_to_goal()

    def _phase(self) -> float:
        t = int(getattr(self._env, "_step", 0)) * int(self._env.frame_skip) * float(
            self._env.model.opt.timestep)
        return 2.0 * np.pi * self._gait.freq * t

    def _abduction_vertices(self) -> list[int]:
        """Hypergraph vertices of the 4 hip-abduction actuators (child body b → vertex b-1)."""
        return [int(self._env.model.jnt_bodyid[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_abduct_{leg}")]) - 1
            for leg in LEGS]

    def _obs(self) -> np.ndarray:
        if self.cfg.obs_mode == "hypergraph":
            return self._obs_hypergraph()
        if self.cfg.obs_mode == "leg_hypergraph":
            return self._obs_leg_hypergraph()
        env = self._env
        dist = float(env.dist_to_goal())
        herr = float(heading_error(env))
        vx = float(env.data.cvel[env.torso, 3])
        vy = float(env.data.cvel[env.torso, 4])
        wz = float(env.data.cvel[env.torso, 2])
        ph = self._phase()
        obs = np.array([dist, np.cos(herr), np.sin(herr), vx, vy, wz,
                        np.sin(ph), np.cos(ph), float(env.data.xmat[env.torso].reshape(3, 3)[2, 2])],
                       dtype=np.float32)
        return self.mirror_obs(obs) if getattr(self, "_mirror", False) else obs

    def _obs_hypergraph(self) -> np.ndarray:
        """Per-vertex ``(n_vertices, 4)`` obs on the body's kinematic hypergraph, for structure
        propagation (signedkan): native ``[qpos, qvel]`` + per-leg gait phase + a GLOBAL lateral
        goal-demand. The signed hyperedges route the lateral demand to the per-leg abduction with the
        structure's signs — so a symmetric crab is representable via weight-sharing, unlike a flat MLP.
        """
        env = self._env
        nf = np.asarray(env.node_features(), np.float32)          # (N, 2): torso [dx, fwd_vel]; leg [qpos, qvel]
        out = np.zeros((nf.shape[0], 4), np.float32)
        out[:, :2] = nf
        ph = self._phase()
        for leg, vtx in enumerate(self._abd_vtx):                 # per-leg gait phase on the abduction vertices
            out[vtx, 2] = float(np.sin(ph + self._phase_pat[leg]))
        herr, dist = float(heading_error(env)), float(env.dist_to_goal())
        out[:, 3] = float(np.clip(np.sin(herr) * dist, -1.0, 1.0))  # GLOBAL signed lateral goal-demand
        return out

    def _obs_leg_hypergraph(self) -> np.ndarray:
        """Minimal ``(5, 4)`` per-vertex obs on the leg-only hypergraph: torso vertex carries the goal
        (forward + lateral) + body velocity; each leg vertex carries its abduction state + gait phase +
        the shared lateral demand. The signed hyperedges route the torso's lateral goal to the legs;
        the per-node weight-sharing makes the crab symmetric across the left/right legs."""
        env = self._env
        herr, dist = float(heading_error(env)), float(env.dist_to_goal())
        lat = float(np.clip(np.sin(herr) * dist, -1.0, 1.0))
        fwd = float(np.cos(herr) * dist)
        out = np.zeros((5, 4), np.float32)
        out[0] = [fwd, lat, float(env.data.cvel[env.torso, 3]), float(env.data.cvel[env.torso, 2])]
        ph = self._phase()
        for leg in range(4):
            out[leg + 1] = [float(env.data.qpos[self._env._leg_qadr[3 * leg]]),
                            float(env.data.qvel[self._abd_dof[leg]]),
                            float(np.sin(ph + self._phase_pat[leg])), lat]
        return out

    # -- gym-like API ----------------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._env.reset(seed=seed if seed is not None else int(self._rng.integers(1 << 30)))
        self._sample_goal()
        self._prev_dist = float(self._env.dist_to_goal())
        self._step_i = 0
        self._mirror = bool(self.cfg.mirror_augment and self._rng.random() < 0.5)  # mirrored episode?
        return self._obs(), {}

    @staticmethod
    def mirror_obs(o: np.ndarray) -> np.ndarray:
        """Left-right mirror of the flat obs [dist, cos(herr), sin(herr), vx, vy, wz, sin(ph), cos(ph), up]:
        flip the lateral goal (sin herr), lateral+yaw velocity, AND the gait phase by π (sin/cos ph) — the
        π-shift the diagonal trot needs to be left-right symmetric."""
        m = np.asarray(o, np.float32).copy()
        m[[2, 4, 5, 6, 7]] *= -1.0
        return m

    @staticmethod
    def mirror_act(a: np.ndarray) -> np.ndarray:
        """Left-right mirror of the 4-D omni abduction [fl,fr,bl,br] → [−fr,−fl,−br,−bl] (swap sides + sign)."""
        a = np.asarray(a, np.float64)
        return -a[[1, 0, 3, 2]]

    def blend_action(self, base: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Bounded residual over the scaffold: ``clip(base + scale·clip(residual, ±1), ±1)``.

        Postcondition: the applied action never departs from ``base`` by more than ``residual_scale``
        per component (before the outer clip) — the safe scaffold is preserved (coin-R8).
        """
        r = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
        return np.clip(base + self.cfg.residual_scale * r, -1.0, 1.0)

    def phase_gates(self) -> np.ndarray:
        """Per-leg phase gate ``g_l = ½(1 + sin(ph + DIAG_PHASE_l)) ∈ [0, 1]`` — synced to each leg's
        trot stride, so a gated residual pulses in phase with the gait (preserving the limit cycle)."""
        ph = self._phase()
        return np.array([0.5 * (1.0 + np.sin(ph + self._phase_pat[leg])) for leg in range(4)],
                        dtype=np.float64)

    def _apply(self, residual: np.ndarray) -> None:
        """Compose the residual with the gait per mode and step the underlying env.

        - ``leg``  : a raw 12-dim residual on the gait's leg targets (``blend_action``) — breaks phase.
        - ``steer``: a 2-dim ``(Δyaw, Δdrive)`` residual on the gait's STEERING + SPEED *parameters*.
        - ``phase``: a 12-dim residual GATED per leg by its trot phase (``phase_gates``) — it only acts
          on each leg in sync with that leg's stride, so it can bias differential stance thrust to
          steer while leaving the periodic limit cycle intact.
        """
        env = self._env
        dist = float(env.dist_to_goal())
        pursuit = float(np.clip(1.1 * float(heading_error(env)), -0.6, 0.6))
        base_drive = 0.0 if dist <= self.cfg.reach_radius else 1.0
        r = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
        if self.cfg.residual_mode == "steer":
            yaw = float(np.clip(pursuit + self.cfg.yaw_res_scale * r[0], -0.6, 0.6))
            drive = float(np.clip(base_drive + self.cfg.drive_res_scale * r[1], 0.0, 1.5))
            final = self._gov.govern(env, self._gait.action(env, yaw_cmd=yaw, drive=drive))
        elif self.cfg.residual_mode == "omni":
            base = self._gov.govern(env, self._gait.action(env, yaw_cmd=pursuit, drive=base_drive))
            ph = self._phase()
            final = base.copy()
            for leg in range(4):                          # per-leg abduction, phase-locked -> lateral crab
                idx = 3 * leg                             # abduction is action index 3*leg+0
                lateral = self.cfg.abd_scale * r[leg] * np.sin(ph + self._phase_pat[leg])
                final[idx] = float(np.clip(final[idx] + lateral, -1.0, 1.0))
        else:
            base = self._gov.govern(env, self._gait.action(env, yaw_cmd=pursuit, drive=base_drive))
            if self.cfg.residual_mode == "phase":
                r = (r.reshape(4, 3) * self.phase_gates()[:, None]).reshape(-1)   # gate per leg by stride phase
            final = self.blend_action(base, r)
        env.step(final)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if getattr(self, "_mirror", False):
            residual = self.mirror_act(residual)          # policy acts in the mirrored view; un-mirror to apply
        self._apply(residual)
        self._step_i += 1
        dist = float(self._env.dist_to_goal())
        upright = float(self._env.data.xmat[self._env.torso].reshape(3, 3)[2, 2])
        herr = float(heading_error(self._env))
        reached = dist <= self.cfg.reach_radius
        fell = upright < self.cfg.fall_upright
        progress = self._prev_dist - dist
        self._prev_dist = dist
        reward = (self.cfg.progress_w * progress
                  + self.cfg.heading_w * float(np.cos(herr))
                  - self.cfg.ctrl_w * float(np.sum(residual ** 2)))
        if reached:
            reward += self.cfg.reach_bonus
        if fell:
            reward -= 5.0
        term = reached or fell
        trunc = self._step_i >= self.cfg.max_steps
        info = {"dist": dist, "reached": reached, "upright": upright, "fell": fell}
        return self._obs(), float(reward), bool(term), bool(trunc), info

    # -- evaluation helpers ----------------------------------------------------
    def rollout_min_dist(self, act_fn, goal: tuple[float, float], seed: int,
                         horizon: int | None = None) -> tuple[float, bool, float]:
        """Roll ``act_fn`` toward a FIXED (dist, bearing_deg) goal; return (min_dist, reached, min_upright)."""
        d, bdeg = goal
        self._mirror = False                              # eval is always the un-mirrored real task
        self._env.reset(seed=seed)
        b = bdeg * np.pi / 180.0
        tx = float(self._env.data.xpos[self._env.torso, 0])
        ty = float(self._env.data.xpos[self._env.torso, 1])
        self._env.goal = np.array([tx + d * np.cos(b), ty + d * np.sin(b)], np.float32)
        self._env._prev_dist = self._env.dist_to_goal()
        self._prev_dist = float(self._env.dist_to_goal())
        self._step_i = 0
        min_dist, min_up = self._prev_dist, 1.0
        for _ in range(horizon or self.cfg.max_steps):
            self._apply(act_fn(self._obs()))
            min_dist = min(min_dist, float(self._env.dist_to_goal()))
            min_up = min(min_up, float(self._env.data.xmat[self._env.torso].reshape(3, 3)[2, 2]))
            if float(self._env.dist_to_goal()) <= self.cfg.reach_radius:
                break
        return round(min_dist, 4), min_dist <= self.cfg.reach_radius, round(min_up, 3)
