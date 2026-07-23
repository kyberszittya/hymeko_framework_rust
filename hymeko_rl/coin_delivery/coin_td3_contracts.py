"""TD3 contracts for PHASE_SWITCHED_LEARNED_LATE_CONTROLLER_V1 (§8) — data structures + invariants only, NO training.

Provides: twin Q critics over the FULL action; full-action target-policy smoothing; n-step returns with n∈{4,8} and
``terminated``/``truncated`` stored and masked separately (terminated ⇒ no bootstrap; truncated ⇒ bootstrap); temporally
COHERENT exploration noise held for 2–4 steps (no per-step independent high-variance noise); and a frozen TD3 config
(delayed actor updates, critic-first warm-up, checkpointed calibration). Nothing here trains — the campaign is gated on
these contracts passing + being committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

ACTION_SCALE = 4.0


# ── twin Q critics over the full action ──
def _qnet(obs_dim: int, action_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(obs_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class LateTwinCritic(nn.Module):
    """TD3 twin critics ``Q_i(obs_48, full_action_4)`` with independent parameters."""

    def __init__(self, obs_dim: int = 48, action_dim: int = 4):
        super().__init__()
        self.q1 = _qnet(obs_dim, action_dim)
        self.q2 = _qnet(obs_dim, action_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


def td3_target_action(pi_late_target, obs_next: torch.Tensor, *, smoothing_std: float, smoothing_clip: float,
                      action_scale: float = ACTION_SCALE, generator: "torch.Generator | None" = None) -> torch.Tensor:
    """TD3 target action with FULL-ACTION target-policy smoothing (§8):

        a' = clip( pi_late_target(obs_next) + clamp(N(0, std), -clip, +clip),  -scale, +scale )

    ``std``/``clip`` are in full-action units (NOT residual units). ``pi_late_target`` is the frozen-per-step target
    actor; it is evaluated under ``no_grad``. # Postcondition: output ∈ [-scale, scale] elementwise."""
    with torch.no_grad():
        base = torch.clamp(pi_late_target.action_mean(obs_next), -action_scale, action_scale)
    noise = torch.randn(base.shape, generator=generator, dtype=base.dtype) * float(smoothing_std)
    eps = torch.clamp(noise, -float(smoothing_clip), float(smoothing_clip))
    return torch.clamp(base + eps, -action_scale, action_scale)


# ── n-step returns with separate terminated/truncated masking ──
def nstep_return(traj, t: int, n: int, gamma: float):
    """From transition ``t`` of one trajectory, accumulate up to ``n`` rewards. Each transition is a dict with
    ``reward``, ``terminated``, ``truncated``, ``obs_next``. Returns ``(reward_sum, bootstrap_obs, mask, gamma_pow)``:

    - stops early on ``terminated`` (mask=0 ⇒ NO bootstrap) or ``truncated`` (mask=1 ⇒ bootstrap, artificial cutoff)
      or on running off the stored trajectory;
    - ``gamma_pow = gamma**steps_accumulated``; the target is ``reward_sum + gamma_pow * mask * Q(bootstrap_obs, a')``.
    """
    G, disc, mask = 0.0, 1.0, 1
    boot = traj[t]["obs_next"]
    steps = 0
    for i in range(n):
        idx = t + i
        if idx >= len(traj):
            break
        tr = traj[idx]
        G += disc * float(tr["reward"]); boot = tr["obs_next"]; disc *= gamma; steps += 1
        if tr["terminated"]:
            mask = 0; break
        if tr["truncated"]:
            mask = 1; break
    return G, boot, mask, disc


@dataclass
class LateReplayBuffer:
    """Trajectory-structured replay for n-step TD3. Stores ``terminated`` and ``truncated`` SEPARATELY (never merged)."""

    trajectories: list = field(default_factory=list)

    def add_trajectory(self, transitions: list) -> None:
        for tr in transitions:
            assert {"obs", "action", "reward", "obs_next", "terminated", "truncated"} <= set(tr), "transition missing keys"
        self.trajectories.append(transitions)

    def n_transitions(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def sample_nstep(self, batch: int, n: int, gamma: float, rng: np.random.Generator):
        """Sample ``batch`` (obs, action, nstep_reward, bootstrap_obs, mask, gamma_pow) tuples."""
        flat = [(ti, t) for ti, traj in enumerate(self.trajectories) for t in range(len(traj))]
        if not flat:
            raise ValueError("empty replay buffer")
        pick = rng.integers(0, len(flat), batch)
        obs, act, rew, boot, mask, gp = [], [], [], [], [], []
        for j in pick:
            ti, t = flat[j]; traj = self.trajectories[ti]
            G, b, m, gpow = nstep_return(traj, t, n, gamma)
            obs.append(traj[t]["obs"]); act.append(traj[t]["action"]); rew.append(G)
            boot.append(b); mask.append(m); gp.append(gpow)
        return (np.stack(obs).astype(np.float32), np.stack(act).astype(np.float32), np.asarray(rew, np.float32),
                np.stack(boot).astype(np.float32), np.asarray(mask, np.float32), np.asarray(gp, np.float32))


# ── temporally coherent exploration noise (held for 2–4 steps) ──
class CoherentNoise:
    """Exploration noise HELD for a random ``hold_len ∈ [hold_min, hold_max]`` steps, then resampled (§8: no per-step
    independent high-variance exploration). Stateful per-episode; call :meth:`reset` at episode start."""

    def __init__(self, action_dim: int = 4, std: float = 0.3, hold_min: int = 2, hold_max: int = 4,
                 action_scale: float = ACTION_SCALE, seed: int = 0):
        assert hold_min >= 2 and hold_max <= 4 and hold_min <= hold_max, "hold length must lie in [2,4]"
        self.action_dim, self.std = int(action_dim), float(std)
        self.hold_min, self.hold_max = int(hold_min), int(hold_max)
        self.action_scale = float(action_scale)
        self.rng = np.random.default_rng(seed)
        self._noise = np.zeros(action_dim, np.float32); self._left = 0; self._hold_len = 0

    def reset(self) -> None:
        self._left = 0

    def sample(self) -> np.ndarray:
        if self._left <= 0:
            self._noise = self.rng.normal(0.0, self.std, self.action_dim).astype(np.float32)
            self._hold_len = int(self.rng.integers(self.hold_min, self.hold_max + 1)); self._left = self._hold_len
        self._left -= 1
        return self._noise.copy()

    def perturb(self, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(action, np.float32) + self.sample(), -self.action_scale, self.action_scale).astype(np.float32)


# Control-only historical config (mechanically-scaled TD3 defaults). NOT the primary; recorded so a preregistered
# smoothing ablation can reference it. The known-good handoff basin is fragile — target smoothing is for LOCAL critic
# regularization, not broad exploration — so the primary uses small full-action smoothing (0.10 / 0.25).
HISTORICAL_SCALED_DEFAULT_SMOOTHING = {"smoothing_std": 0.2 * ACTION_SCALE, "smoothing_clip": 0.5 * ACTION_SCALE,
                                       "status": "control-only; do not run unless a preregistered ablation requires it"}


# ── frozen TD3 config ──
@dataclass(frozen=True)
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    n_step_set: tuple = (4, 8)                          # frozen small set
    policy_delay: int = 2                               # delayed actor updates
    smoothing_std: float = 0.10                         # PRIMARY, full-action units (local critic regularization)
    smoothing_clip: float = 0.25                        # PRIMARY, full-action units
    exploration_std_init: float = 0.15                  # coherent-noise std at start (full-action units)
    exploration_std_max: float = 0.30                   # coherent-noise std ceiling
    coherent_noise_hold: tuple = (2, 4)                 # held 2–4 steps
    critic_warmup_steps: int = 2000                     # critic-first warm-up before actor updates
    checkpoints: tuple = (0, 1000, 3000, 6000, 10000, 20000, 40000)
    grad_clip: float = 1.0

    def frozen_manifest(self) -> dict:
        return {"gamma": self.gamma, "tau": self.tau, "batch_size": self.batch_size,
                "actor_lr": self.actor_lr, "critic_lr": self.critic_lr, "n_step_set": list(self.n_step_set),
                "policy_delay": self.policy_delay, "smoothing_std_full_action": self.smoothing_std,
                "smoothing_clip_full_action": self.smoothing_clip,
                "exploration_std_init": self.exploration_std_init, "exploration_std_max": self.exploration_std_max,
                "coherent_noise_hold": list(self.coherent_noise_hold), "critic_warmup_steps": self.critic_warmup_steps,
                "checkpoints": list(self.checkpoints), "grad_clip": self.grad_clip,
                "historical_scaled_default_smoothing": HISTORICAL_SCALED_DEFAULT_SMOOTHING,
                "note": "PRIMARY target smoothing 0.10/0.25 (full-action, local regularization — NOT the scaled 0.8/2.0); "
                        "exploration is SEPARATE (coherent noise, std 0.15→0.30, held 2–4 steps); "
                        "terminated/truncated masked separately"}
