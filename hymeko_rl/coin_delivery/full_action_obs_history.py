"""FULL_ACTION_OBS_HISTORY_V1 — the history-aware learner input for the receding-horizon feedback pilot (§2).

The frozen canonical observation ``node_features`` (48) carries coin POSITION but no coin VELOCITY, so the settle
sub-task is a POMDP for a reactive policy (measured 2026-07-22: reactive and frame-stack-k=3 BCs both cap ~1/9). This
contract exposes the missing state WITHOUT touching the physical robot, semantic graph, reward, success predicate, or
action contract — it is a pure LEARNER-side windowing of the same canonical observation + the actor's own executed
actions.

Layout (newest-first, causal — the feature at step t contains only information available BEFORE choosing action t):

    feature_t = [ obs_t (48) | obs_{t-1} (48) | obs_{t-2} (48) | a_{t-1} (4) | a_{t-2} (4) ]   -> 152

Deterministic padding at episode start: missing observations = the initial observation; missing actions = zeros.
The k=3 observation window exposes coin velocity, contact-transition direction, target-entry-vs-exit motion, and the
settling trend as finite differences the actor can read.

The legacy 48-value instantaneous observation stays available for historical checkpoint reproduction, but it is NOT
Markov-complete for the new full-action actor.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np

K_OBS = 3
K_ACT = 2
BASE_OBS_DIM = 48
ACTION_DIM = 4
FEATURE_DIM = K_OBS * BASE_OBS_DIM + K_ACT * ACTION_DIM   # 152
SEMANTIC_GRAPH_FP = "sem:469094de1fba54b2ff481706ca2e09ce"
BUNDLE_HASH = "6664ac459cca8f62"


class ObsHistoryV1:
    """Maintains the newest-first observation + action windows and emits the 152-dim causal feature. Deterministic:
    the same (initial obs, action/obs stream) always yields the same features; reset re-pads."""

    def __init__(self) -> None:
        self._obs: deque = deque(maxlen=K_OBS)
        self._act: deque = deque(maxlen=K_ACT)

    def reset(self, initial_obs: np.ndarray) -> None:
        o = np.asarray(initial_obs, np.float32).flatten()
        self._obs.clear()
        self._act.clear()
        for _ in range(K_OBS):
            self._obs.appendleft(o.copy())                   # pad with the initial observation
        for _ in range(K_ACT):
            self._act.appendleft(np.zeros(ACTION_DIM, np.float32))

    def feature(self) -> np.ndarray:
        """[obs_t, obs_{t-1}, obs_{t-2}, a_{t-1}, a_{t-2}] — only pre-decision information."""
        return np.concatenate(list(self._obs) + list(self._act)).astype(np.float32)

    def push(self, next_obs: np.ndarray, executed_action: np.ndarray) -> None:
        """Advance one step: record the action just executed and the resulting observation (newest-first)."""
        self._act.appendleft(np.asarray(executed_action, np.float32).flatten()[:ACTION_DIM].copy())
        self._obs.appendleft(np.asarray(next_obs, np.float32).flatten().copy())


def build_history_features(obs: np.ndarray, act: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """Vectorised, per-trajectory (never crossing a boundary) construction of the causal 152-dim features that MATCH
    :class:`ObsHistoryV1` exactly: ``feature[j] = [obs_j, obs_{j-1}, obs_{j-2}, act_{j-1}, act_{j-2}]`` with obs padded
    by the trajectory's first frame and actions padded by zeros. Label for row j stays ``act[j]``."""
    n = len(obs)
    out = np.empty((n, FEATURE_DIM), np.float32)
    for tid in np.unique(traj):
        idx = np.where(traj == tid)[0]                       # contiguous + time-ordered
        o, a = obs[idx], act[idx]
        for j in range(len(idx)):
            ob = [o[j], o[j - 1] if j >= 1 else o[0], o[j - 2] if j >= 2 else o[0]]
            ac = [a[j - 1] if j >= 1 else np.zeros(ACTION_DIM, np.float32),
                  a[j - 2] if j >= 2 else np.zeros(ACTION_DIM, np.float32)]
            out[idx[j]] = np.concatenate(ob + ac)
    return out


def contract_spec() -> dict:
    """Serialisable contract definition + its SHA-256 (recorded in every pilot manifest)."""
    spec = {
        "name": "FULL_ACTION_OBS_HISTORY_V1",
        "k_obs": K_OBS, "k_act": K_ACT,
        "base_obs_dim": BASE_OBS_DIM, "action_dim": ACTION_DIM, "feature_dim": FEATURE_DIM,
        "layout": "[obs_t, obs_{t-1}, obs_{t-2}, a_{t-1}, a_{t-2}] newest-first, causal",
        "padding": "obs -> initial observation; actions -> zeros; deterministic at episode start",
        "semantic_graph_fp": SEMANTIC_GRAPH_FP, "bundle_hash": BUNDLE_HASH,
        "reset_behaviour": "obs window filled with initial obs (x3); action window zeros (x2)",
        "legacy_markov_complete": False,
    }
    payload = json.dumps(spec, sort_keys=True).encode()
    spec["sha256"] = hashlib.sha256(payload).hexdigest()
    return spec


if __name__ == "__main__":
    print(json.dumps(contract_spec(), indent=1))
