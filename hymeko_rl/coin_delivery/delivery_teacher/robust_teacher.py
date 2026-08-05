"""R11.5R robust delivery teacher: re-certify each scenario's delivery with a CEM that scores LEXICOGRAPHICALLY for local
K6-survival (wide basins), not just nominal K6. ONLY the teacher scoring changes — the CEM structure (pop/iters/elite), the
action-coordinate, the physics, strict K6, and safety are frozen (mirrored from ``solver._search``).

The lexicographic key is: safety -> nominal K6 -> survival@0.5% -> survival@1% -> -CVaR(dtz) -> survival@2% -> -time. A
candidate is survival-eligible ONLY if nominal K6 AND safe, so a lucky perturbation can never rescue a failed nominal.
Survival uses a SHARED, reproducible perturbation seed-bank so the nominal (T0) and robust (T1) teachers are comparable, and
a progressive funnel (screen 0.5% -> refine 1% -> stress 2%) keeps it economical.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO, clip_theta
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.theta_option.delivery_theta_env import COIN_HARD, QDOT_HARD
from hymeko_rl.coin_delivery.theta_option.robust_delivery import WideBasinDeliveryCertificate

_SPAN = THETA_HI - THETA_LO
SCREEN, REFINE, STRESS = 0.005, 0.01, 0.02       # the funnel scales (0.5% / 1% / 2%)
_INELIGIBLE_CVAR = -1e9                           # ineligible candidates sort below every eligible one


@dataclass(frozen=True)
class RobustTeacherConfig:
    k_screen: int = 3            # 0.5% screening rollouts (every nominal-K6 candidate)
    k_refine: int = 5            # 1% refine rollouts (only screen survivors)
    k_stress: int = 6            # 2% stress rollouts (only refine survivors)
    screen_min: float = 0.34     # survival@0.5% needed to escalate to 1% (a loose funnel gate)
    cvar_alpha: float = 0.5      # penalize the mean of the worst alpha-fraction of perturbation dtz
    robust_min_survival: float = 0.75   # survival@1% for WIDE_RECERTIFIED status


class PerturbationBank:
    """Shared, reproducible delta directions per scale — the SAME deltas evaluate every theta, so survival is comparable
    across T0/T1 and across scenarios at a given rank."""

    def __init__(self, scales: "tuple[float, ...]", k_max: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._d = {sc: rng.standard_normal((k_max, THETA_LO.shape[0])) for sc in scales}

    def deltas(self, scale: float, k: int) -> np.ndarray:
        return self._d[scale][:k]


def _safe(m: dict) -> bool:
    return bool(m["peak_qdot"] <= QDOT_HARD and m["peak_coin_speed"] <= COIN_HARD)


def survival_at(snap: Any, theta: np.ndarray, scale: float, k: int, bank: PerturbationBank,
                alpha: float) -> "tuple[float, float, int]":
    """(K6-survival fraction, CVaR worst-tail dtz mm, #rollouts) over the shared bank deltas at ``scale``."""
    ms = [rollout_primitive(snap, clip_theta(theta + scale * _SPAN * d), CLOSED_LOOP_CFG) for d in bank.deltas(scale, k)]
    surv = round(sum(delivery_success(m, CLOSED_LOOP_CFG) for m in ms) / k, 3) if k else 0.0
    dtzs = sorted((float(m["dtz_end"]) * 1000 for m in ms), reverse=True)
    cvar = round(float(np.mean(dtzs[: max(1, int(round(alpha * k)))])), 2) if dtzs else 0.0
    return surv, cvar, len(ms)


def robust_evaluate(snap: Any, theta: np.ndarray, cfg: RobustTeacherConfig, bank: PerturbationBank
                    ) -> "tuple[tuple, dict, int]":
    """Lexicographic key (higher is better), a record, and the number of perturbation rollouts spent (funnel)."""
    m = rollout_primitive(snap, clip_theta(theta), CLOSED_LOOP_CFG)
    nom_k6, safe = bool(delivery_success(m, CLOSED_LOOP_CFG)), _safe(m)
    dtz = round(float(m["dtz_end"]) * 1000, 2)
    if not (nom_k6 and safe):
        return (int(safe), int(nom_k6), 0.0, 0.0, _INELIGIBLE_CVAR, 0.0, 0.0), \
            {"nom_k6": nom_k6, "safe": safe, "surv05": 0.0, "surv1": 0.0, "surv2": 0.0, "cvar_dtz": dtz, "dtz_mm": dtz}, 0
    s05, cvar, n0 = survival_at(snap, theta, SCREEN, cfg.k_screen, bank, cfg.cvar_alpha)
    s1 = s2 = 0.0
    n1 = n2 = 0
    if s05 >= cfg.screen_min:
        s1, cvar, n1 = survival_at(snap, theta, REFINE, cfg.k_refine, bank, cfg.cvar_alpha)
        if s1 > 0.0:
            s2, _c, n2 = survival_at(snap, theta, STRESS, cfg.k_stress, bank, cfg.cvar_alpha)
    key = (int(safe), int(nom_k6), s05, s1, -cvar, s2, -float(m["release_step"]))    # time as tie-break (energy is R11.8)
    return key, {"nom_k6": True, "safe": True, "surv05": s05, "surv1": s1, "surv2": s2, "cvar_dtz": cvar, "dtz_mm": dtz}, \
        n0 + n1 + n2


@dataclass(frozen=True)
class RobustSolveResult:
    theta: "tuple[float, ...]"
    record: dict
    compute: dict

    def certificate(self) -> WideBasinDeliveryCertificate:
        r = self.record
        return WideBasinDeliveryCertificate(bool(r["nom_k6"]), REFINE, float(r["surv1"]), float(r["cvar_dtz"]),
                                            bool(r["safe"]))


def robust_cem(snap: Any, spec: Any, cfg: RobustTeacherConfig, bank: PerturbationBank, seed: int) -> RobustSolveResult:
    """CEM over the frozen search structure (``spec`` pop/iters/elite) but ranked by the lexicographic robust key (no
    early-exit — keep searching for higher survival even after K6 is reached). Accounts nominal vs perturbation rollouts."""
    rng = np.random.default_rng(seed)
    dim = len(spec.search_idx)
    mean = np.array([(spec.lo[j] + spec.hi[j]) / 2.0 for j in range(dim)], np.float64)
    std = np.array(spec.init_std, np.float64)
    best: "tuple[tuple, np.ndarray, dict] | None" = None
    proposals = nom_rolls = pert_rolls = 0
    for _ in range(spec.iters):
        pop = mean[None] + std[None] * rng.standard_normal((spec.pop, dim))
        scored = []
        for x in pop:
            key, rec, n_pert = robust_evaluate(snap, spec.assemble(x), cfg, bank)
            proposals += 1
            nom_rolls += 1
            pert_rolls += n_pert
            scored.append((key, x))
            if best is None or key > best[0]:
                best = (key, spec.assemble(x), rec)
        scored.sort(key=lambda z: z[0], reverse=True)
        elite = np.stack([x for _, x in scored[: spec.elite]])
        mean, std = elite.mean(0), elite.std(0) * 0.9 + 1e-3
    assert best is not None
    return RobustSolveResult(tuple(round(float(t), 6) for t in best[1]), best[2],
                             {"proposals": proposals, "nom_rollouts": nom_rolls, "pert_rollouts": pert_rolls,
                              "sim_calls": nom_rolls + pert_rolls})


def recert_status(record: dict, cfg: RobustTeacherConfig) -> str:
    """WIDE_RECERTIFIED (robust theta found) / NARROW_ONLY (nominal K6 but not robust) / NO_NOMINAL_K6."""
    if not record.get("nom_k6"):
        return "NO_NOMINAL_K6"
    if bool(record.get("safe")) and float(record.get("surv1", 0.0)) >= cfg.robust_min_survival:
        return "WIDE_RECERTIFIED"
    return "NARROW_ONLY"
