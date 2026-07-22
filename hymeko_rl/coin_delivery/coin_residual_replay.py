"""Replay / controller-state contract V2 + TD3 target-action construction for the phase-gated learned-residual
controller (§5).

Decisive invariant: **a replay minibatch never reconstructs, advances, or queries a fresh phase FSM.** The gate
multiplier used at collection time is stored per transition (``gate_t``, ``gate_tp1`` + the full
``PHASE_GATE_CONTROLLER_STATE_V2``), and the TD3 target action for transition ``t`` is built from the *stored*
``gate_tp1`` — the learner holds no gate object.

Target-action contract (§5.3):

    base_tp1     = clip(pi_0(obs_tp1), -4, 4)                        # frozen, NEVER smoothed
    residual_tp1 = clip(0.25*tanh(residual_target(obs_tp1)) + eps, -0.25, 0.25)   # smoothing on the residual only,
                                                                                  # re-bounded to the residual range
    target_tp1   = clip(base_tp1 + gate_tp1 * residual_tp1, -4, 4)

When ``gate_tp1 == 0`` the target equals ``pi_0(obs_tp1)`` bit-identically and the smoothing noise has no effect.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_residual_controller import RESIDUAL_BOUND

ACTION_SCALE = 4.0

CONTROLLER_STATE_V2_SCHEMA = {
    "schema": "PHASE_GATE_CONTROLLER_STATE_V2",
    "gate_values_allowed": [0.0, 1.0],
    "fields": ["gate", "mode", "bilateral_counter", "uni_counter", "uni_side", "loss_counter"],
    "provenance_fields": ["coin_hist", "ltip_hist", "rtip_hist", "last_arm_mechanism", "comotion_ok"],
    "note": "gate is the deployed multiplier used at collection time; no phase inference from obs during training",
}


def controller_state_schema_hash() -> str:
    return hashlib.sha256(json.dumps(CONTROLLER_STATE_V2_SCHEMA, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ReplayControllerStateV2:
    """The controller state actually produced online, stored by value in replay. ``gate`` is the load-bearing field
    for target construction; the rest is causal provenance to reproduce the transition."""
    gate: float
    mode: str
    bilateral_counter: int = 0
    uni_counter: int = 0
    uni_side: "str | None" = None
    loss_counter: int = 0
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.gate) not in (0.0, 1.0):
            raise ValueError(f"gate must be in the deployed contract {{0,1}}, got {self.gate!r}")

    @classmethod
    def from_gate(cls, gate) -> "ReplayControllerStateV2":
        """Snapshot a live :class:`StableEngagementGate` by VALUE (deep-copied provenance)."""
        st = gate.state_v2()
        return cls(gate=float(gate.gate), mode=str(st["mode"]), bilateral_counter=int(st["bilateral_counter"]),
                   uni_counter=int(st["uni_counter"]), uni_side=st["uni_side"], loss_counter=int(st["loss_counter"]),
                   provenance={k: json.loads(json.dumps(st[k])) for k in
                               ("coin_hist", "ltip_hist", "rtip_hist", "last_arm_mechanism", "comotion_ok")})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResidualTransition:
    obs_t: np.ndarray
    action_t: np.ndarray
    reward_t: float
    obs_tp1: np.ndarray
    done_t: float
    cstate_t: ReplayControllerStateV2
    cstate_tp1: ReplayControllerStateV2


class ResidualReplayBuffer:
    """Stores complete residual transitions with V2 controller states, BY VALUE. Sampling returns tensors including
    the stored ``gate_t``/``gate_tp1``; it NEVER advances a gate FSM."""

    def __init__(self) -> None:
        self._obs, self._act, self._rew, self._obs2, self._done = [], [], [], [], []
        self._gate_t, self._gate_tp1 = [], []
        self._cstate_t, self._cstate_tp1 = [], []

    def __len__(self) -> int:
        return len(self._obs)

    def add(self, tr: ResidualTransition) -> None:
        self._obs.append(np.array(tr.obs_t, np.float32))          # copies (defensive)
        self._act.append(np.array(tr.action_t, np.float32))
        self._rew.append(float(tr.reward_t))
        self._obs2.append(np.array(tr.obs_tp1, np.float32))
        self._done.append(float(tr.done_t))
        self._gate_t.append(float(tr.cstate_t.gate))
        self._gate_tp1.append(float(tr.cstate_tp1.gate))
        self._cstate_t.append(tr.cstate_t.to_dict())             # frozen dataclass -> plain dict snapshot
        self._cstate_tp1.append(tr.cstate_tp1.to_dict())

    def sample(self, idx) -> dict:
        idx = np.asarray(idx)
        return {"obs": torch.tensor(np.stack([self._obs[i] for i in idx])),
                "action": torch.tensor(np.stack([self._act[i] for i in idx])),
                "reward": torch.tensor(np.array([self._rew[i] for i in idx], np.float32)),
                "obs2": torch.tensor(np.stack([self._obs2[i] for i in idx])),
                "done": torch.tensor(np.array([self._done[i] for i in idx], np.float32)),
                "gate_t": torch.tensor(np.array([self._gate_t[i] for i in idx], np.float32)),
                "gate_tp1": torch.tensor(np.array([self._gate_tp1[i] for i in idx], np.float32)),
                "cstate_t": [self._cstate_t[i] for i in idx], "cstate_tp1": [self._cstate_tp1[i] for i in idx]}


def bounded_smoothed_residual(residual_target, obs_tp1: torch.Tensor, *, noise: "torch.Tensor | None" = None,
                              smoothing_std: float = 0.2, smoothing_clip: float = 0.5,
                              bound: float = RESIDUAL_BOUND) -> torch.Tensor:
    """0.25*tanh(raw) + clamped target-policy smoothing, re-bounded to [-bound, bound] (§5.3). Smoothing applies to
    the RESIDUAL only; the result never exceeds the permitted residual range."""
    residual = bound * torch.tanh(residual_target.raw(obs_tp1))
    if noise is None:
        noise = torch.randn_like(residual) * smoothing_std
    eps = torch.clamp(noise, -smoothing_clip, smoothing_clip)
    return torch.clamp(residual + eps, -bound, bound)


def residual_target_action(pi0, residual_target, obs_tp1: torch.Tensor, gate_tp1: torch.Tensor, *,
                           noise: "torch.Tensor | None" = None, smoothing_std: float = 0.2,
                           smoothing_clip: float = 0.5, bound: float = RESIDUAL_BOUND) -> torch.Tensor:
    """TD3 target action from the FROZEN base + gated smoothed residual, using the STORED ``gate_tp1`` (§5.3).

    # Preconditions: ``gate_tp1`` values are in {0,1} (the deployed contract). # Postconditions: rows with
    ``gate_tp1==0`` equal ``clip(pi_0(obs_tp1))`` bit-identically; the base branch is never smoothed. # Invariants:
    no gate FSM is instantiated or advanced; ``pi_0`` receives no gradient (evaluated under ``no_grad``)."""
    with torch.no_grad():
        base = torch.clamp(pi0.action_mean(obs_tp1), -ACTION_SCALE, ACTION_SCALE)     # frozen, no smoothing
    residual = bounded_smoothed_residual(residual_target, obs_tp1, noise=noise, smoothing_std=smoothing_std,
                                         smoothing_clip=smoothing_clip, bound=bound)
    return torch.clamp(base + gate_tp1.unsqueeze(-1) * residual, -ACTION_SCALE, ACTION_SCALE)


def td_target_scalar(reward: torch.Tensor, done: torch.Tensor, gamma: float, q_next: torch.Tensor) -> torch.Tensor:
    """y = reward + gamma*(1-done)*q_next — terminal transitions mask the bootstrap (§5.4 test 6)."""
    return reward + gamma * (1.0 - done) * q_next
