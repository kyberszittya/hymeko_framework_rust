"""R11.6A reward-driven coin delivery RL — an ``option_rl.OptionEnv`` over the certified-handoff bank whose action is the
FULL structured transport theta (push/ramp/release/brake), and a shaped physical reward over the rollout measurements.

The policy learns to deliver the coin to the (variable) target at strict K6 WITHOUT per-instance CEM and WITHOUT imitating
any teacher theta — the 56 certified teacher theta are warm-start / positive reference only. This is a genuinely new
configuration vs the prior coin-RL walls: full structured theta (not a bounded residual over a frozen base, R8/R9),
multi-scenario + target-conditioned + shaped reward (not a sigma=0.05 ball around one scaffold, R10.2), and 56 certified
demonstrations across 51 scenarios (not the 2 dev cradles that gated R7 off). TD3, not SAC (R8: TD3 >> SAC).

Energy stays DIAGNOSTIC per the frozen R11 energy contract (objective at R11.8); efficiency is carried by the time and
overshoot terms. The transport primitive, the K6 certificate, and the certified box are read-only here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO, Standardizer, clip_theta
from hymeko_rl.coin_delivery.forward_displacement import HELD_DWELL, delivery_success, rollout_primitive

QDOT_HARD = 3.0            # MotionLimits.joint_vel_hard — the frozen safety gate (also enforced inside the primitive)
COIN_HARD = 1.5            # MotionLimits.ee_speed_hard
_HORIZON = CLOSED_LOOP_CFG.horizon


def box_to_theta(z: np.ndarray) -> np.ndarray:
    """Map a tanh-bounded action ``z in [-1,1]^6`` to a clipped certified transport theta."""
    z = np.clip(np.asarray(z, np.float64), -1.0, 1.0)
    return clip_theta(THETA_LO + 0.5 * (z + 1.0) * (THETA_HI - THETA_LO))


def theta_to_box(theta: np.ndarray) -> np.ndarray:
    """Inverse of ``box_to_theta`` (for warm-starting the actor at a teacher theta)."""
    theta = clip_theta(theta)
    return np.clip(2.0 * (theta - THETA_LO) / (THETA_HI - THETA_LO) - 1.0, -1.0, 1.0)


@dataclass(frozen=True)
class DeliveryReward:
    """Shaped physical reward over the rollout measurements ``m`` (template: r9_delivery_train.DeliveryReward).

    Safety is a HARD barrier (not a tunable weight) mirroring ``_feasible``/``delivery_success``; strict K6 is the dominant
    terminal term and the unchanged external certificate; progress/overshoot/time shape the interior. Energy is NOT read
    (frozen R11 energy contract → R11.8). Deterministic pure function of ``m``.
    """

    w_gap: float = 6.0
    w_forward: float = 2.0
    w_dwell: float = 2.0
    w_k6: float = 10.0
    w_overshoot: float = 3.0
    w_cross: float = 2.0
    w_grasp: float = 4.0
    w_time: float = 1.0
    safety_barrier: float = 20.0

    def __call__(self, m: "dict[str, Any]") -> float:
        if float(m["peak_qdot"]) > QDOT_HARD or float(m["peak_coin_speed"]) > COIN_HARD:
            return -self.safety_barrier                                     # hard feasibility barrier
        forward = max(0.0, float(m["forward"]))
        r = (self.w_gap * float(m["gap_closed"])
             + self.w_forward * min(forward / 0.10, 1.0)
             + self.w_dwell * float(m["k6_max_dwell"]) / HELD_DWELL
             + self.w_k6 * float(bool(m["k6_delivered"]))
             - self.w_overshoot * float(m["terminal_coin_speed"])
             - self.w_cross * float(m["cross"]) / max(forward, 1e-3)
             - self.w_grasp * min(float(m["lost_before_release"]), 10.0) / 10.0
             - self.w_time * float(m["release_step"]) / _HORIZON)
        return float(r)


@dataclass
class ScenarioHandoff:
    """One certified-handoff training example: the reconstructed snapshot (cached), its conditioning descriptor, and the
    teacher theta (warm-start reference). ``snap`` is self-contained — ``rollout_primitive`` branches it per episode."""

    scenario_id: str
    split: str
    seed: int
    snap: Any
    descriptor: np.ndarray
    teacher_theta: np.ndarray


class CoinDeliveryThetaOptionEnv:
    """``option_rl.OptionEnv`` over the certified-handoff bank. Terminal one-step option: one structured theta per episode.

    Preconditions: ``handoffs`` are certified post-capture snapshots targeting each scenario's zone; ``standardizer`` is
    fit on the training descriptors. Postconditions: ``step`` returns a terminal transition whose ``info`` carries
    ``tau, terminal, k6, safe, dtz_mm`` for the semi-MDP trainer + selection metric.
    """

    def __init__(self, handoffs: "list[ScenarioHandoff]", standardizer: Standardizer,
                 reward: DeliveryReward = DeliveryReward(), seed: int = 0) -> None:
        assert handoffs, "empty handoff bank"
        self._h = handoffs
        self._std = standardizer
        self._reward = reward
        self._rng = np.random.default_rng(seed)
        self._train_idx = [i for i, h in enumerate(handoffs) if h.split == "train"] or list(range(len(handoffs)))
        self._cur = self._train_idx[0]
        self.obs_dim = int(handoffs[0].descriptor.shape[0])
        self.act_dim = 6

    def _obs(self, idx: int) -> np.ndarray:
        return self._std.transform(self._h[idx].descriptor).astype(np.float32)

    def reset(self, idx: "int | None" = None) -> np.ndarray:
        """No ``idx`` → a random TRAIN scenario (dev/test are reserved for evaluation); explicit ``idx`` for eval."""
        self._cur = int(self._rng.choice(self._train_idx)) if idx is None else int(idx)
        return self._obs(self._cur)

    def indices(self, split: str) -> "list[int]":
        return [i for i, h in enumerate(self._h) if h.split == split]

    def teacher_action(self, idx: int) -> np.ndarray:
        """Scenario ``idx``'s teacher theta in box-normalized action space (for immutable positive-replay seeding)."""
        return theta_to_box(self._h[idx].teacher_theta).astype(np.float32)

    def warmstart_data(self) -> "tuple[np.ndarray, np.ndarray]":
        """(standardized train obs, teacher theta in box-normalized action space) for BC-initializing the actor —
        the teacher is a warm-start reference, not an imitation target the policy must keep."""
        idx = self.indices("train")
        x = np.array([self._obs(i) for i in idx], np.float32)
        y = np.array([theta_to_box(self._h[i].teacher_theta) for i in idx], np.float32)
        return x, y

    def step(self, action: np.ndarray, *, search_seed: "int | None" = None) -> "tuple[np.ndarray, float, bool, dict]":
        h = self._h[self._cur]
        theta = box_to_theta(action)
        m = rollout_primitive(h.snap, theta, CLOSED_LOOP_CFG)
        reward = self._reward(m)
        k6 = bool(delivery_success(m, CLOSED_LOOP_CFG))
        safe = bool(m["peak_qdot"] <= QDOT_HARD and m["peak_coin_speed"] <= COIN_HARD)
        info = {"tau": float(_HORIZON), "terminal": 1.0, "end": "completed", "k6": float(k6), "safe": safe,
                "dtz_mm": round(float(m["dtz_end"]) * 1000, 2), "scenario_id": h.scenario_id, "split": h.split}
        return self._obs(self._cur), reward, True, info


def fit_obs_standardizer(handoffs: "list[ScenarioHandoff]") -> Standardizer:
    """Standardizer fit on the TRAIN-split descriptors only (never dev/test — keeps the held-out honest)."""
    train = [h.descriptor for h in handoffs if h.split == "train"]
    return Standardizer.fit(np.array(train if train else [h.descriptor for h in handoffs], np.float64))
