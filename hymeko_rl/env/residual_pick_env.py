"""Residual-action wrapper for the FANUC stable-place env — RL learns a BOUNDED correction on a frozen base policy.

The naive RL setup (raw joint-position action, ±π joints + [0,0.035] grip, un-normalised) is the coin-toss
`v14c SAC/TD3 fail=action-space` trap: a huge, non-uniform action space and RL replacing (not refining) a strong
imitation policy → drift off the good manifold. This wrapper fixes both:

- the RL action is a **normalised residual** ``r ∈ [-1,1]^A`` (uniform — SAC/PPO see a clean action space);
- the executed action is ``clip(base(obs) + delta_scale·r, lo, hi)`` — a *bounded* correction on the frozen base
  policy (the 0.83 hybrid-DAgger policy). Zero residual reproduces the base, so RL starts competent and can only
  refine within ``delta_scale`` — the coin-toss "bounded option-RL POSITIVE" regime, not the "unbounded residual
  destructive" one.

The wrapper is a gym-5-tuple env with the SAME observation/hypergraph as the underlying env, so the existing
trainers/actors build on it unchanged. The reward + termination are the underlying stable-place env's.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box

from hymeko_rl.control.hybrid_automaton import HybridAutomaton, PickState
from hymeko_rl.viz.render_pick_place import fanuc_pick_env


class ResidualPickEnv:
    """Frozen-base + learned-residual action wrapper.

    # Preconditions ``base`` has ``action_mean(obs_tensor) -> action`` (an ``ActorCritic``/SAC actor); the
      underlying env is a ``require_settle`` FANUC pick env. ``reward_mode ∈ {"base","settle"}``. # Invariants the
      base policy is frozen (eval, no grad); the executed action stays within the underlying action bounds; the
      observation/termination are the underlying env's — only the reward is remapped when ``reward_mode="settle"``.
    """

    def __init__(self, base: "torch.nn.Module", *, delta_scale: "np.ndarray | float" = 0.25,
                 grip_scale: float = 0.02, reward_mode: str = "base", time_cost: float = 0.02,
                 settle_bonus: float = 20.0, center_bonus: float = 20.0, center_shape: float = 100.0,
                 active_modes: "tuple[str, ...] | None" = None, env_kwargs: "dict | None" = None) -> None:
        if reward_mode not in ("base", "settle", "precision"):
            raise ValueError(f"reward_mode must be 'base'|'settle'|'precision', got {reward_mode!r}")
        self.env = fanuc_pick_env(**(env_kwargs or dict(expert_version=3, require_settle=True, max_steps=1000)))
        self._reward_mode = reward_mode
        self._time_cost = float(time_cost)
        self._settle_bonus = float(settle_bonus)
        self._center_bonus = float(center_bonus)
        self._center_shape = float(center_shape)
        self._place_radius = float(self.env.place_radius)
        self._prev_err: "float | None" = None            # potential-shaping state (object→centre distance, m)
        # Mode-gating: RL authority only in the named hybrid-automaton modes (STRUCTURED residual — the coin-toss
        # "structured primitive learning WORKED" regime). Elsewhere the residual is zeroed → the reliable base runs.
        self._active_modes = tuple(active_modes) if active_modes else None
        self._auto = HybridAutomaton.from_hymeko() if self._active_modes else None
        self.base = base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        lo = np.asarray(self.env.action_space.low, dtype=np.float32)
        hi = np.asarray(self.env.action_space.high, dtype=np.float32)
        self._lo, self._hi = lo, hi
        ds = np.full(lo.shape, float(delta_scale), dtype=np.float32) if np.isscalar(delta_scale) \
            else np.asarray(delta_scale, dtype=np.float32)
        ds[-1] = float(grip_scale)                                # grip has a tiny range → its own residual scale
        self._delta = ds
        self.action_space = Box(-1.0, 1.0, shape=lo.shape, dtype=np.float32)   # normalised residual
        self.observation_space = self.env.observation_space
        self._obs = None

    # --- delegate the graph/task surface the trainers/actors read ---
    @property
    def hg(self) -> Any:
        return self.env.hg

    @property
    def n_actions(self) -> int:
        return int(self.env.n_actions)

    def node_features(self) -> np.ndarray:
        return self.env.node_features()

    @property
    def max_steps(self) -> int:
        return int(self.env.max_steps)

    @property
    def data(self) -> Any:                       # expose the sim so metrics/renderers (LiftPlaceMetric divergence
        return self.env.data                     # guard reads env.data.qacc) work transparently through the wrapper

    @property
    def model(self) -> Any:
        return self.env.model

    def _base_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.base.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()

    def reset(self, *, seed: "int | None" = None, options: "dict | None" = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        self._prev_err = None                            # reset potential-shaping state each episode
        return obs, info

    def _reward(self, base_rew: float, info: "dict[str, Any]") -> float:
        """Reward the residual optimises. ``base`` = the underlying env's dense reward. ``settle`` = an anti-farm
        reward keyed on the SAME event the eval counts (``info['reached']`` = ``placed_stable``): a per-step time
        cost + a terminal settle bonus + the env's table-clearance penalty. It removes the dense contact bonus
        (``+0.5·(left+right)``/step) that, over the 1000-step horizon, pays a residual to HOLD rather than release —
        i.e. to inflate return by NOT stable-placing. Under ``settle`` the only way to score is to settle, sooner is
        better, and hitting the table still costs — perfectly aligned with :class:`LiftPlaceMetric`.

        ``precision`` = ``settle`` PLUS (a) a **dense potential-based** centering term
        ``center_shape·(prev_err − err)`` — farm-proof (Ng-potential shaping telescopes to endpoints, so it cannot
        be gamed by hovering) that rewards reducing the object→centre distance *every* step, giving PPO a real
        gradient; and (b) a terminal centering bonus ``center_bonus·(1 − err/place_radius)`` at the settle. Binary
        success is still required for the settle bonus (non-farmable), but now the residual is driven to place the
        object at the target CENTRE (base ≈ 4.7 cm off) — the headroom that binary stable-place lacks."""
        if self._reward_mode == "base":
            return base_rew
        pen = self.env.ground_penalty if info.get("approach_contact") else 0.0
        r = -self._time_cost - pen
        if info.get("reached"):
            r += self._settle_bonus
        if self._reward_mode == "precision":
            err = float(info["obj_to_target"])
            if self._prev_err is not None:
                r += self._center_shape * (self._prev_err - err)                 # dense: reward reducing centre-dist
            self._prev_err = err
            if info.get("reached"):
                center = 1.0 - min(err / self._place_radius, 1.0)                # 1=centre, 0=edge
                r += self._center_bonus * center
        return r

    def _gate(self, r: np.ndarray) -> "tuple[np.ndarray, str, bool]":
        """Zero the residual outside the active hybrid-automaton modes. Returns ``(r, mode, active)``."""
        if self._auto is None:
            return r, "all", True
        mode = self._auto.classify(PickState.from_env(self.env))
        active = mode in self._active_modes
        return (r if active else np.zeros_like(r)), mode, active

    def step(self, residual: np.ndarray):
        base = self._base_action(self._obs)
        r = np.clip(np.asarray(residual, dtype=np.float32), -1.0, 1.0)
        r, mode, active = self._gate(r)                                       # structured: RL only in active modes
        action = np.clip(base + self._delta * r, self._lo, self._hi)          # bounded correction on the base
        obs, rew, term, trunc, info = self.env.step(action)
        self._obs = obs
        info = {**info, "residual_norm": float(np.linalg.norm(r)), "mode": mode, "residual_active": active}
        return obs, self._reward(float(rew), info), term, trunc, info


class ResidualActionAdapter:
    """Wrap a residual-policy actor so it acts on the UNDERLYING env (base + residual) — for evaluation on the
    real stable-place env with :func:`eval_success`. ``action_mean`` returns the executed (base+residual) action."""

    def __init__(self, base: "torch.nn.Module", residual: "torch.nn.Module", *,
                 delta_scale: "np.ndarray | float" = 0.25, grip_scale: float = 0.02, action_dim: int = 7,
                 lo: "np.ndarray | None" = None, hi: "np.ndarray | None" = None) -> None:
        self.base, self.residual = base.eval(), residual.eval()
        ds = np.full(action_dim, float(delta_scale), dtype=np.float32) if np.isscalar(delta_scale) \
            else np.asarray(delta_scale, dtype=np.float32)
        ds[-1] = float(grip_scale)
        self._delta = torch.as_tensor(ds)
        self._lo = torch.as_tensor(lo) if lo is not None else None
        self._hi = torch.as_tensor(hi) if hi is not None else None

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.base.action_mean(obs)
            r = torch.clamp(self.residual.action_mean(obs), -1.0, 1.0)
            a = base + self._delta.to(base.device) * r
            if self._lo is not None:
                a = torch.clamp(a, self._lo.to(a.device), self._hi.to(a.device))
            return a

    def parameters(self):                                        # eval_success only reads action_mean; harmless
        return self.residual.parameters()
