"""R11.6B robustness-seeking generalization — the ONE variable that changes vs R11.6A v2.1: the reward semantics.

Rewarding nominal K6 admits narrow, knife-edge theta (R11.4B/R11.6A). Rewarding K6 that SURVIVES local theta-perturbation
steers RL toward WIDE-basin solutions that generalize to new coin/target geometries. Everything else (structured theta,
BC warm-start, TD3, positive replay, splits, safety + strict-K6 certificate, combined selection) stays frozen.

The robust rollout (1 nominal + K perturbation rollouts, perturbation in NORMALIZED theta-box coordinate) is packaged as a
HyMeKo ``WideBasinDeliveryCertificate`` so RL chases an interpreted control property (delivers the coin AND does so in a
locally stable way), not a raw number. A nominally-failed theta earns NO survival credit (the 1_K6(nominal) gate), so a
lucky perturbation cannot reward a bad nominal.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO, clip_theta
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.theta_option.delivery_theta_env import (
    COIN_HARD,
    QDOT_HARD,
    CoinDeliveryThetaOptionEnv,
    DeliveryReward,
    box_to_theta,
)

_HORIZON = CLOSED_LOOP_CFG.horizon
_SPAN = THETA_HI - THETA_LO


@dataclass(frozen=True)
class RobustRewardConfig:
    """K perturbation rollouts at relative box scale ``sigma`` (smoke 3/0.5%, full 4-5/1%); CVaR over the worst tail."""

    k: int = 4
    sigma: float = 0.01
    lambda_surv: float = 6.0
    lambda_tail: float = 2.0
    cvar_alpha: float = 0.5       # penalize the mean of the worst alpha-fraction of perturbation dtz
    robust_min_survival: float = 0.75    # a scenario is "robust" iff nominal K6 AND survival >= this (dev-robust gate)


@dataclass(frozen=True)
class WideBasinDeliveryCertificate:
    """A HyMeKo delivery-robustness certificate: not just 'reaches the target' but 'reaches it locally stably'."""

    nominal_k6: bool
    perturbation_scale: float
    survival_rate: float
    worst_dtz_mm: float
    safe: bool

    def is_robust(self, min_survival: float) -> bool:
        return bool(self.nominal_k6 and self.safe and self.survival_rate >= min_survival)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def perturb_theta(theta: np.ndarray, sigma: float, rng: Any) -> np.ndarray:
    """Perturb in the NORMALIZED box coordinate (sigma * box-range), so large-scale params do not dominate; clipped."""
    return clip_theta(np.asarray(theta, np.float64) + sigma * _SPAN * rng.standard_normal(theta.shape[0]))


def robust_rollout(snap: Any, theta: np.ndarray, cfg: RobustRewardConfig, rng: Any) -> "tuple[dict, WideBasinDeliveryCertificate]":
    """1 nominal + K perturbation rollouts; returns (nominal measurements, certificate). Deterministic given ``rng``."""
    theta = clip_theta(theta)
    m_nom = rollout_primitive(snap, theta, CLOSED_LOOP_CFG)
    nominal_k6 = bool(delivery_success(m_nom, CLOSED_LOOP_CFG))
    nominal_safe = bool(m_nom["peak_qdot"] <= QDOT_HARD and m_nom["peak_coin_speed"] <= COIN_HARD)
    pert = [rollout_primitive(snap, perturb_theta(theta, cfg.sigma, rng), CLOSED_LOOP_CFG) for _ in range(cfg.k)]
    survival = round(sum(delivery_success(m, CLOSED_LOOP_CFG) for m in pert) / cfg.k, 3) if cfg.k else 0.0
    dtzs = sorted((float(m["dtz_end"]) * 1000 for m in pert), reverse=True)
    n_tail = max(1, int(round(cfg.cvar_alpha * cfg.k)))
    worst = round(float(np.mean(dtzs[:n_tail])), 2) if dtzs else round(float(m_nom["dtz_end"]) * 1000, 2)
    return m_nom, WideBasinDeliveryCertificate(nominal_k6, cfg.sigma, survival, worst, nominal_safe)


class RobustDeliveryReward:
    """Lexicographic surrogate: safety barrier -> nominal task reward -> survival (gated on nominal K6) -> CVaR(dtz)."""

    def __init__(self, task_reward: DeliveryReward, cfg: RobustRewardConfig) -> None:
        self._task = task_reward
        self._cfg = cfg

    def __call__(self, m_nom: dict, cert: WideBasinDeliveryCertificate) -> float:
        if float(m_nom["peak_qdot"]) > QDOT_HARD or float(m_nom["peak_coin_speed"]) > COIN_HARD:
            return -self._task.safety_barrier
        base = self._task(m_nom)                                             # progress + nominal K6 - overshoot/time
        survival = self._cfg.lambda_surv * (1.0 if cert.nominal_k6 else 0.0) * cert.survival_rate
        tail = self._cfg.lambda_tail * cert.worst_dtz_mm / 100.0             # mm -> O(1); steer away from narrow basins
        return float(base + survival - tail)


class RobustCoinDeliveryEnv(CoinDeliveryThetaOptionEnv):
    """The v2.1 env with the ONLY change being robust reward semantics: ``step`` does 1 nominal + K perturbation rollouts
    and returns the robust reward; the certificate's survival rides in ``info``. Reset/state/action are unchanged."""

    def __init__(self, handoffs: list, standardizer: Any, task_reward: DeliveryReward, robust_cfg: RobustRewardConfig,
                 seed: int = 0) -> None:
        super().__init__(handoffs, standardizer, task_reward, seed)
        self._rcfg = robust_cfg
        self._robust = RobustDeliveryReward(task_reward, robust_cfg)
        self._pert_rng = np.random.default_rng(seed + 100003)               # stochastic robust reward during training

    def step(self, action: np.ndarray, *, search_seed: "int | None" = None) -> "tuple[np.ndarray, float, bool, dict]":
        h = self._h[self._cur]
        m_nom, cert = robust_rollout(h.snap, box_to_theta(action), self._rcfg, self._pert_rng)
        info = {"tau": float(_HORIZON), "terminal": 1.0, "end": "completed", "k6": float(cert.nominal_k6),
                "safe": cert.safe, "dtz_mm": round(float(m_nom["dtz_end"]) * 1000, 2), "survival": cert.survival_rate,
                "robust": float(cert.is_robust(self._rcfg.robust_min_survival)), "scenario_id": h.scenario_id,
                "split": h.split}
        return self._obs(self._cur), self._robust(m_nom, cert), True, info

    def certify(self, theta: np.ndarray, idx: int, seed: int = 0) -> WideBasinDeliveryCertificate:
        """Fixed-seed robust certificate for scenario ``idx`` (reproducible eval / dev-robust gate / test panel)."""
        _m, cert = robust_rollout(self._h[idx].snap, theta, self._rcfg, np.random.default_rng(seed))
        return cert
