"""Composite-action twin critic conditioned on the encoded ``ReplayControllerStateV2`` (§6).

The critic evaluates the *deployed composite action* ``clip(pi_0(o) + g*r(o), -4, 4)`` — never the residual alone —
and is conditioned on an explicit, deterministic encoding of the controller state, because the same physical action
has different meaning under EARLY_CONTROL vs LATE_CONTROL_ARMED vs REACQUIRE. No fresh FSM, no phase inference from
the observation, no target/success/planner/future leakage: the encoder reads only stored causal fields.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn

# fixed categorical vocabularies (documented, versioned)
_MODES = ("EARLY_CONTROL", "LATE_CONTROL_ARMED", "REACQUIRE", "TERMINAL")
_SIDES = ("L", "R", None)
_BILATERAL_CAP, _UNI_CAP, _LOSS_CAP = 10.0, 20.0, 5.0     # normalization caps for the counters

CONTROLLER_STATE_ENCODER_V1 = {
    "encoder": "PHASE_GATE_CONTROLLER_STATE_ENCODER_V1",
    "field_order": ["gate", "mode_onehot(EARLY,ARMED,REACQUIRE,TERMINAL)", "bilateral_counter/10",
                    "uni_counter/20", "loss_counter/5", "side_onehot(L,R,None)"],
    "dim": 1 + len(_MODES) + 3 + len(_SIDES),            # 1 + 4 + 3 + 3 = 11
    "note": "deterministic; only stored causal fields; no strings enter the net; no target/success/future leakage",
}
ENCODER_DIM = CONTROLLER_STATE_ENCODER_V1["dim"]


def encoder_fingerprint() -> str:
    return hashlib.sha256(json.dumps(CONTROLLER_STATE_ENCODER_V1, sort_keys=True).encode()).hexdigest()


def encode_controller_state(cstate: dict) -> np.ndarray:
    """Deterministically encode one stored controller state dict → fixed-length float vector (no FSM, no strings)."""
    mode = str(cstate.get("mode", "EARLY_CONTROL"))
    side = cstate.get("uni_side", None)
    vec = np.zeros(ENCODER_DIM, np.float32)
    vec[0] = float(cstate.get("gate", 0.0))
    vec[1 + _MODES.index(mode) if mode in _MODES else 1] = 1.0
    off = 1 + len(_MODES)
    vec[off + 0] = min(float(cstate.get("bilateral_counter", 0)) / _BILATERAL_CAP, 1.0)
    vec[off + 1] = min(float(cstate.get("uni_counter", 0)) / _UNI_CAP, 1.0)
    vec[off + 2] = min(float(cstate.get("loss_counter", 0)) / _LOSS_CAP, 1.0)
    soff = off + 3
    vec[soff + (_SIDES.index(side) if side in _SIDES else 2)] = 1.0
    return vec


def encode_controller_states(cstates: "list[dict]") -> torch.Tensor:
    return torch.tensor(np.stack([encode_controller_state(c) for c in cstates]))


def _qnet(obs_dim: int, action_dim: int) -> nn.Sequential:
    d = obs_dim + action_dim + ENCODER_DIM
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class CompositeTwinCritic(nn.Module):
    """TD3 twin critics ``Q_i(obs, composite_action, encoded_controller_state)`` with INDEPENDENT parameters."""

    def __init__(self, obs_dim: int = 48, action_dim: int = 4):
        super().__init__()
        self.q1 = _qnet(obs_dim, action_dim)
        self.q2 = _qnet(obs_dim, action_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, enc_state: torch.Tensor):
        x = torch.cat([obs, action, enc_state], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, obs, action, enc_state):
        q1, q2 = self.forward(obs, action, enc_state)
        return torch.min(q1, q2)

    def contract(self) -> dict:
        return {"critic": "CompositeTwinCritic", "input": "concat(obs 48, composite_action 4, encoded_state 11)",
                "encoder": CONTROLLER_STATE_ENCODER_V1["encoder"], "encoder_dim": ENCODER_DIM,
                "twin": "independent Q1/Q2", "action": "deployed clipped composite action (never residual-only)"}

    def contract_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.contract(), sort_keys=True).encode()).hexdigest()
