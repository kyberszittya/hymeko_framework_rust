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

import numpy as np

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .locomotion_gait import _DIAG_PHASE, SteeredTrotGait, heading_error
from .motion_contract import JointVelocityGovernor


@dataclass(frozen=True)
class ResidualTrotConfig:
    """Task distribution + reward shaping for the residual-over-trot goal-reaching problem."""

    dist_lo: float = 0.5
    dist_hi: float = 0.75
    bearing_deg: float = 40.0            # goals sampled in bearing ∈ [−bearing_deg, +bearing_deg]
    residual_mode: str = "leg"           # "leg" = 12-dim raw-target residual | "steer" = 2-dim (Δyaw, Δdrive) gait-param residual | "phase" = 12-dim residual PHASE-GATED per leg | "omni" = 4-dim per-leg ABDUCTION amplitude (phase-locked lateral crab over the forward trot — the RICHER action space, adds lateral DOF the trot leaves unused)
    abd_scale: float = 0.5               # omni mode: bound on the learned per-leg abduction (lateral) amplitude
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

    def __init__(self, dim: int, low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
        self.shape = (dim,)
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
        self._gait = SteeredTrotGait()
        self._gov = JointVelocityGovernor(v_max=self.cfg.v_max)
        self._rng = np.random.default_rng(self.seed)
        self._prev_dist = 0.0
        self._step_i = 0
        _dims = {"leg": 12, "phase": 12, "steer": 2, "omni": 4}
        act_dim = _dims[self.cfg.residual_mode]
        self.action_space = _Box(act_dim, seed=self.seed)
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

    def _obs(self) -> np.ndarray:
        env = self._env
        dist = float(env.dist_to_goal())
        herr = float(heading_error(env))
        vx = float(env.data.cvel[env.torso, 3])
        vy = float(env.data.cvel[env.torso, 4])
        wz = float(env.data.cvel[env.torso, 2])
        ph = self._phase()
        return np.array([dist, np.cos(herr), np.sin(herr), vx, vy, wz,
                         np.sin(ph), np.cos(ph), float(env.data.xmat[env.torso].reshape(3, 3)[2, 2])],
                        dtype=np.float32)

    # -- gym-like API ----------------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._env.reset(seed=seed if seed is not None else int(self._rng.integers(1 << 30)))
        self._sample_goal()
        self._prev_dist = float(self._env.dist_to_goal())
        self._step_i = 0
        return self._obs(), {}

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
        return np.array([0.5 * (1.0 + np.sin(ph + _DIAG_PHASE[leg])) for leg in range(4)],
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
                lateral = self.cfg.abd_scale * r[leg] * np.sin(ph + _DIAG_PHASE[leg])
                final[idx] = float(np.clip(final[idx] + lateral, -1.0, 1.0))
        else:
            base = self._gov.govern(env, self._gait.action(env, yaw_cmd=pursuit, drive=base_drive))
            if self.cfg.residual_mode == "phase":
                r = (r.reshape(4, 3) * self.phase_gates()[:, None]).reshape(-1)   # gate per leg by stride phase
            final = self.blend_action(base, r)
        env.step(final)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
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
