"""RESIDUAL_CRITIC_STATE_V2 — the causal state fed to the composite/advantage CRITIC (only).

    critic_state = [ FULL_ACTION_OBS_HISTORY_V1 (152) | encode(PHASE_GATE_CONTROLLER_STATE_V2) (11) ]  -> 163

The 152-dim history recovers coin velocity (absent from the instantaneous 48-dim observation), which transport /
contact-retention / settling need. The frozen base actor pi_0 is NOT touched — it keeps its original 48-dim
``node_features``. Only the critic and advantage critic receive this augmentation. No future observation and no
offline phase label enter the state (the encoder reads only stored causal gate fields).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from hymeko_rl.coin_delivery.coin_residual_critic import ENCODER_DIM, encode_controller_state
from hymeko_rl.coin_delivery.full_action_obs_history import FEATURE_DIM, ObsHistoryV1, build_history_features
from hymeko_rl.coin_delivery.full_action_obs_history import contract_spec as _hist_contract

RESIDUAL_CRITIC_STATE_DIM = FEATURE_DIM + ENCODER_DIM      # 152 + 11 = 163


def residual_critic_state_v2_contract() -> dict:
    spec = {"name": "RESIDUAL_CRITIC_STATE_V2", "dim": RESIDUAL_CRITIC_STATE_DIM,
            "components": {"history": _hist_contract()["name"], "history_dim": FEATURE_DIM,
                          "gate_encoding": "PHASE_GATE_CONTROLLER_STATE_ENCODER_V1", "gate_dim": ENCODER_DIM},
            "history_sha": _hist_contract()["sha256"][:16],
            "layout": "concat(history_feature 152, encode_controller_state(gate_state_v2) 11)",
            "base_actor_untouched": "pi_0 receives its original 48-dim node_features; only the critic sees V2",
            "no_future_obs": True, "no_offline_phase_label": True}
    spec["sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


class ResidualCriticStateV2:
    """Streaming causal critic-state. ``feature(gate_state_dict)`` returns the 163-dim vector for the current step.

    # Preconditions: ``reset`` called with the episode's initial observation; ``push`` called after each executed
      action with the resulting observation. # Postconditions: the history half equals :func:`build_history_features`
      (streaming==batch); only pre-decision information is used. # Invariants: no future observation enters the state.
    """

    def __init__(self) -> None:
        self._hist = ObsHistoryV1()

    def reset(self, initial_obs: np.ndarray) -> None:
        self._hist.reset(initial_obs)

    def feature(self, gate_state: dict) -> np.ndarray:
        return np.concatenate([self._hist.feature(), encode_controller_state(gate_state)]).astype(np.float32)

    def push(self, next_obs: np.ndarray, executed_action: np.ndarray) -> None:
        self._hist.push(next_obs, executed_action)

    # ---- resume (checkpoint reproduces history_t+1) ----
    def state_dict(self) -> dict:
        return {"obs": [o.tolist() for o in self._hist._obs], "act": [a.tolist() for a in self._hist._act]}

    def load_state_dict(self, sd: dict) -> None:
        from collections import deque

        from hymeko_rl.coin_delivery.full_action_obs_history import K_ACT, K_OBS
        self._hist._obs = deque([np.asarray(o, np.float32) for o in sd["obs"]], maxlen=K_OBS)
        self._hist._act = deque([np.asarray(a, np.float32) for a in sd["act"]], maxlen=K_ACT)


def build_critic_states_v2(obs: np.ndarray, act: np.ndarray, traj: np.ndarray,
                           gate_states: "list[dict]") -> np.ndarray:
    """Batch reconstruction of the 163-dim critic states that MATCH the streaming :class:`ResidualCriticStateV2`
    exactly: ``[build_history_features(obs,act,traj) | encode_controller_state(gate_state)]`` per row."""
    hist = build_history_features(obs, act, traj)
    enc = np.stack([encode_controller_state(g) for g in gate_states])
    return np.concatenate([hist, enc], axis=1).astype(np.float32)
