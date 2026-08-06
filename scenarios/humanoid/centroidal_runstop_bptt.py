r"""Gradient-based RL for run-stop via differentiable simulation (BPTT) — a torch alternative to CEM.

The run-stop dynamics are differentiable, so the cleanest gradient method is **backprop-through-time**: roll a
torch policy through a torch copy of ``runstop_step`` and descend a differentiable surrogate of the objective
(final speed² + a smooth upright-penalty). This is genuine gradient-based policy optimisation (an analytic policy
gradient), and — because the trained net shares the 1-hidden-layer shape of ``policy_actions`` — its weights are
flattened back into the numpy policy so the SAME held-out ``evaluate`` metric compares it to CEM and tuned-linear.

(Model-free TD3 is the method when the simulator is NOT differentiable; here BPTT is exact and cheaper.)

# Preconditions: a ``RunStopConfig``. # Postconditions: ``to_numpy_params`` yields a vector ``policy_actions`` can
#   run, so BPTT and CEM policies are scored by identical held-out stop-success.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scenarios.humanoid.centroidal_runstop import PolicyConfig, RunStopConfig, mixed_set


class BpttPolicy(nn.Module):
    """A 1-hidden-layer tanh MLP (matching ``policy_actions``' shape) → bounded (fx, a)."""

    def __init__(self, cfg: RunStopConfig, hidden: int = 24) -> None:
        super().__init__()
        self.cfg = cfg
        self.lin1 = nn.Linear(5, hidden)
        self.lin2 = nn.Linear(hidden, 2)

    def forward(self, feats: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        out = torch.tanh(self.lin2(torch.tanh(self.lin1(feats))))
        return self.cfg.fx_max * out[:, 0], self.cfg.a_max * out[:, 1]

    def to_numpy_params(self) -> np.ndarray:
        """Flatten to the layout ``policy_actions`` expects: [W1(5,h), b1, W2(h,2), b2] (torch weight is out×in)."""
        return np.concatenate([
            self.lin1.weight.detach().numpy().T.ravel(), self.lin1.bias.detach().numpy(),
            self.lin2.weight.detach().numpy().T.ravel(), self.lin2.bias.detach().numpy()])


def _torch_step(state: torch.Tensor, t: float, fx: torch.Tensor, a: torch.Tensor, cfg: RunStopConfig) -> torch.Tensor:
    """Differentiable copy of ``runstop_step`` (control only in stance). ``state`` columns = (vx, L, pitch)."""
    stance = 1.0 if (t % cfg.cycle) < cfg.ts else 0.0
    fx, a = fx * stance, a * stance
    vx = state[:, 0] + fx * cfg.dt
    ll = state[:, 1] + (-cfg.l_damp * state[:, 1] * stance + a - cfg.k_couple * fx) * cfg.dt
    pitch = state[:, 2] + (ll / cfg.inertia) * cfg.dt
    return torch.stack([vx, ll, pitch], dim=1)


def rollout_loss(policy: BpttPolicy, x0: torch.Tensor, cfg: RunStopConfig, w_pitch: float = 20.0,
                 tail_frac: float = 0.3) -> torch.Tensor:
    r"""Differentiable surrogate aligned with the true objective (stop AND stay stopped, stay upright).

    ``mean_batch [ mean_{tail} vx² + w · mean_t relu(|pitch|−0.75·fall_pitch)² ]``. The **tail** speed penalty (over
    the last ``tail_frac`` of the episode, not just the final step) is what makes BPTT match a direct method: a
    final-step-only ``vx²`` lets the policy dip to zero speed for one instant; penalising the whole tail forces a
    genuine, held stop. (Fixing this surrogate took BPTT from 0.83 to 1.00 held-out stop-success.)
    """
    state, v_run = x0, x0[:, 0]
    steps = int(round(cfg.horizon / cfg.dt))
    t_start = int(steps * (1.0 - tail_frac))
    p_safe = 0.75 * cfg.fall_pitch
    pitch_pen, tail = torch.zeros(len(x0)), torch.zeros(len(x0))
    for i in range(steps):
        t = i * cfg.dt
        targ = torch.clamp(v_run * (1.0 - (t - cfg.t_stop) / cfg.ramp), min=0.0) if t >= cfg.t_stop else v_run
        phase = torch.full((len(x0),), (t % cfg.cycle) / cfg.cycle)
        feats = torch.stack([state[:, 0], state[:, 1], state[:, 2], targ, phase], dim=1)
        fx, a = policy(feats)
        state = _torch_step(state, t, fx, a, cfg)
        pitch_pen = pitch_pen + torch.relu(state[:, 2].abs() - p_safe) ** 2
        if i >= t_start:
            tail = tail + state[:, 0] ** 2
    return (tail / (steps - t_start) + w_pitch * (pitch_pen / steps)).mean()


def train_bptt(cfg: RunStopConfig, iters: int = 500, lr: float = 3e-3, hidden: int = 24, seed: int = 0) -> BpttPolicy:
    """Backprop-through-time policy optimisation (Adam, gradient-clipped) on the differentiable run-stop surrogate."""
    torch.manual_seed(seed)
    policy = BpttPolicy(cfg, hidden)
    x0 = torch.as_tensor(mixed_set(cfg, offset=0.0), dtype=torch.float32)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = rollout_loss(policy, x0, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()
    return policy


def bptt_numpy_params(cfg: RunStopConfig, pc: PolicyConfig, **kw) -> np.ndarray:
    """Train BPTT and return the numpy-policy parameter vector (scored by the shared ``evaluate``)."""
    if pc.hidden != 24:
        raise ValueError("BpttPolicy uses hidden=24 to match policy_actions; set PolicyConfig(hidden=24)")
    return train_bptt(cfg, hidden=pc.hidden, **kw).to_numpy_params()
