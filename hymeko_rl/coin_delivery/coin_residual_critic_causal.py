"""CAUSAL_COMPOSITE_TWIN_CRITIC — the CAUSAL_HISTORY arm of the §8 state-sufficiency ablation.

Twin critics ``Q_i(critic_state_v2, composite_action)`` where ``critic_state_v2`` is the 163-dim
``RESIDUAL_CRITIC_STATE_V2`` = ``[FULL_ACTION_OBS_HISTORY_V1 (152) | encode(PHASE_GATE_CONTROLLER_STATE_V2) (11)]``.
The instantaneous arm (:class:`~hymeko_rl.coin_delivery.coin_residual_critic.CompositeTwinCritic`) sees only the 48-dim
instantaneous observation + the same 11-dim gate encoding; this arm additionally sees the 3-step causal obs history and
the 2 previous executed actions, which recover coin velocity/momentum absent from a single 48-dim frame.

This is a *controlled* twin: identical capacity (256-256), identical optimizer/target-smoothing/schedule; only the
critic-state representation differs. It evaluates the *deployed composite action* — never the residual alone — exactly
as the instantaneous twin does. ``pi_0`` is never fed this state (the frozen base keeps its canonical 48-dim input).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_residual_critic_state import RESIDUAL_CRITIC_STATE_DIM

ACTION_DIM = 4


def _qnet(state_dim: int, action_dim: int) -> nn.Sequential:
    d = state_dim + action_dim
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class CausalCompositeTwinCritic(nn.Module):
    """TD3 twin critics ``Q_i(critic_state_v2 163, composite_action 4)`` with INDEPENDENT parameters.

    # Preconditions: ``state`` rows are 163-dim :data:`RESIDUAL_CRITIC_STATE_DIM` vectors from
      :func:`~hymeko_rl.coin_delivery.coin_residual_critic_state.build_critic_states_v2` (or the streaming
      :class:`ResidualCriticStateV2`); ``action`` rows are the deployed clipped composite action.
    # Postconditions: returns ``(q1, q2)`` scalars per row; ``min_q`` is the conservative Bellman value.
    """

    def __init__(self, state_dim: int = RESIDUAL_CRITIC_STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.q1 = _qnet(self.state_dim, self.action_dim)
        self.q2 = _qnet(self.state_dim, self.action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def contract(self) -> dict:
        return {"critic": "CausalCompositeTwinCritic",
                "input": f"concat(critic_state_v2 {self.state_dim}, composite_action {self.action_dim})",
                "state": "RESIDUAL_CRITIC_STATE_V2 (FULL_ACTION_OBS_HISTORY_V1 152 + gate encoding 11)",
                "twin": "independent Q1/Q2",
                "action": "deployed clipped composite action (never residual-only)"}

    def contract_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.contract(), sort_keys=True).encode()).hexdigest()


def q1_grad_wrt_action(critic: nn.Module, state: np.ndarray, action: np.ndarray,
                       *, causal: bool, enc_state=None) -> np.ndarray:
    """dQ1/d(action) at ``action`` — the load-bearing local ascent direction (§10). Works for both arms:

    - causal:  ``critic(state_163, action)``          (``enc_state`` unused)
    - instant: ``critic(obs_48, action, enc_state)``  (``enc_state`` = encoded gate 11)
    """
    a = torch.tensor(np.asarray(action, np.float32)[None], requires_grad=True)
    s = torch.tensor(np.asarray(state, np.float32)[None])
    if causal:
        q1, _ = critic(s, a)
    else:
        q1, _ = critic(s, a, enc_state)
    q1.backward()
    return a.grad[0].numpy()
