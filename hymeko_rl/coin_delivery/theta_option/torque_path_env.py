"""R10.2 Stage 2 (Boundary 4) — HOME-start structured-option capture environment + K6-dominant reward.

An ``OptionEnv`` over the FROZEN torque-path coordinate: one structured option ``theta`` is decided at READY (reached by
the frozen analytic HOME->READY transit), executed as the deterministic torque-path capture, handed to the frozen
downstream, and scored by strict K6. Every episode is HOME-derived (the initiation READY is the frozen transit output);
the primary reported result is HOME->K6, not READY->K6.

Reward (deliberately simple, no new cathedral; NO cradle-centred tau* shaping):

    R = w_K * 1[strict K6] + w_d * progress(min_dtz) - w_b * P_boundary - w_s * P_safety - w_t * P_tube

with ``w_K >> w_d > w_b, w_t`` and a strong safety penalty ``w_s``. The phase-tube term is a *mild* regulariser measured
relative to the scaffold's own response to the SAME initiation state (``q-q0(phi), qvel-qvel0(phi), prev-tau0(phi)``).

``reset != 1`` handling (Boundary-3 review contract): a downstream that takes a different HANDOFF_RESET count is a
``boundary_route_variation`` — NOT unsafe, NOT a K6 success, NOT eligible for positive replay; it receives a terminal
boundary penalty and counts as a FAILURE in evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo


@dataclass(frozen=True)
class RewardWeights:
    """K6-dominant weights (``w_K >> w_d > w_b, w_t``; strong safety). Frozen; not tuned during training."""

    k6: float = 10.0            # w_K — dominant strict-K6 bonus
    min_dtz: float = 1.0        # w_d — dense min_dtz progress (in [-1, 0])
    boundary: float = 0.5       # w_b — reset != 1 terminal penalty
    safety: float = 20.0        # w_s — strong safety penalty (> w_K)
    tube: float = 0.2           # w_t — mild phase-tube regulariser
    miss_cap_mm: float = 60.0   # progress saturates at this miss distance
    tube_q: float = 1.0
    tube_v: float = 0.25
    tube_tau: float = 0.25
    tube_cap: float = 5.0       # phase-tube penalty is capped so it stays a mild regulariser


def classify(k6: bool, safe: bool, reset: int) -> str:
    """Outcome class. ``unsafe`` (safety) > ``boundary_route_variation`` (reset != 1) > ``k6`` / ``safe_negative``. A
    ``k6`` success requires safe AND exactly one HANDOFF_RESET."""
    if not safe:
        return "unsafe"
    if reset != 1:
        return "boundary_route_variation"
    return "k6" if k6 else "safe_negative"


def _tube_penalty(obs_arr: np.ndarray, tube: dict, w: RewardWeights) -> float:
    """Mean squared deviation of the executed trajectory from the scaffold phase tube (q, qvel, prev), capped."""
    dq = obs_arr[:, 0:4] - tube["q0"]
    dv = obs_arr[:, 4:8] - tube["qvel0"]
    dt = obs_arr[:, 8:12] - tube["tau0"]
    pen = (w.tube_q * float(np.mean(np.sum(dq ** 2, axis=1)))
           + w.tube_v * float(np.mean(np.sum(dv ** 2, axis=1)))
           + w.tube_tau * float(np.mean(np.sum(dt ** 2, axis=1))))
    return float(min(pen, w.tube_cap))


def option_reward(cls: str, min_dtz: float, tube_pen: float, w: RewardWeights) -> float:
    """K6-dominant reward from the outcome class + dense min_dtz progress + mild tube regulariser. No tau* term."""
    return float(w.k6 * (cls == "k6")
                 + w.min_dtz * (-min(min_dtz, w.miss_cap_mm) / w.miss_cap_mm)
                 - w.boundary * (cls == "boundary_route_variation")
                 - w.safety * (cls == "unsafe")
                 - w.tube * tube_pen)


class StructuredOptionCaptureEnv:
    """An ``OptionEnv`` (reset -> READY obs; step(theta) -> one terminal option). The initiation states are HOME-derived
    perturbed-READY snapshots; each ``step`` rolls the deterministic torque-path capture + frozen downstream and returns
    the K6-dominant reward. Exploration (``theta = clip(actor(obs) + sigma*D*z, -1, 1)``) is applied by the training loop,
    not here — the env rolls whatever ``theta`` it is given."""

    def __init__(self, rig: dict, perturbations: "list[crl.Perturbation]", *, weights: RewardWeights = RewardWeights()):
        self.down, self.weights = rig["down"], weights
        self.states: "list[tuple[Any, dict]]" = []
        for pert in perturbations:
            snap = crl.perturb_ready(rig["ready"], rig["stack"], pert)
            roller = tpo.TorquePathCaptureRoll(snap, rig["ref"], rig["stack"], rig["pi0"], rig["coin"])
            self.states.append((roller, tpo.record_phase_tube(roller)))
        self._i = 0
        self._cur = 0

    def __len__(self) -> int:
        return len(self.states)

    def _ready_obs(self, roller: Any) -> np.ndarray:
        rl = roller.cap.ready.branch()
        return crl.capture_observation(rl, roller.cap.prev0.copy(), roller.ref, roller.coin0, 0, roller.pi0.steps)

    def reset(self, idx: "int | None" = None) -> np.ndarray:
        self._cur = (self._i % len(self.states)) if idx is None else idx
        self._i += 1
        return self._ready_obs(self.states[self._cur][0])

    def step(self, action: np.ndarray, *, search_seed: "int | None" = None) -> "tuple[np.ndarray, float, bool, dict]":
        roller, tube = self.states[self._cur]
        res = roller.rollout(np.asarray(action, np.float32))
        k6, md, safe, kinds = self.down.deliver_with_trace(res["snapshot"])
        reset = kinds.count("HANDOFF_RESET")
        cls = classify(bool(k6), bool(safe), reset)
        tube_pen = _tube_penalty(np.asarray(res["obs"]), tube, self.weights)
        reward = option_reward(cls, float(md), tube_pen, self.weights)
        info = {"tau": float(roller.pi0.steps), "terminal": 1.0, "end": "handoff", "state_idx": self._cur,
                "k6": float(cls == "k6"), "min_dtz": float(md), "safe": bool(safe), "reset": reset, "class": cls,
                "boundary_route_variation": cls == "boundary_route_variation", "tube_pen": tube_pen, "reward": reward}
        return self._ready_obs(roller), reward, True, info
