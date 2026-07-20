"""COIN-DELIVERY new control primitives — Track B centering/settling primitive (+ its CEM oracle).

COIN-DELIVERY-RL-2 (F-COIN-DELIVERY-RL2) split the hard states into an ACQUISITION wall (never grasp / lose grasp) and
a PRECISION wall (ZONE_ONLY / NEAR_MISS reach ~0.034 but a bounded residual around grasp_carry can't tighten to the
0.02 centre). This module builds a NEW centering primitive for the precision wall — NOT a residual around sustained
grasp_carry, but a distinct control law:

  * PROPORTIONAL midpoint correction toward the zone centre: translation magnitude ∝ distance, so it decelerates as it
    approaches (grasp_carry translates at constant full speed and OVERSHOOTS/oscillates near the centre);
  * contact-preserving squeeze while transporting;
  * a settle mode inside a deadband near the centre (gentle micro-motion + settle squeeze) so the coin comes to rest
    within center_tol instead of being pushed back out.

The primitive plugs into the existing phase-conditioned harness via ``CoinDeliveryTrainEnv._base_override`` (grasp_carry
still runs the acquisition prefix; the centering primitive runs the transport/centering suffix with zero residual). Its
CEM oracle sweeps the 5 primitive parameters to maximise center-reach on the precision-failure states — the Track-B
gate for whether off-policy RL on this subproblem is justified. NO RL here, NO env/reward/dynamics/CORE change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.experiments.coin_delivery1 import _dir_to_zone
from hymeko_rl.train.coin_delivery_rl import CoinDeliveryTrainEnv, DeliveryRLConfig, make_delivery_rl_env, roll_delivery, scripted_action_fn

# CEM-searched parameter ranges (normalised [0,1] → these), documented so the primitive is legible:
_RANGES = (("kp", 3.0, 40.0),        # proportional gain (translation ∝ kp·distance, clipped to 1)
           ("sq_hold", 0.2, 1.0),    # transport squeeze (maintain grasp)
           ("sq_settle", -0.3, 0.8), # settle-mode squeeze (may ease off to let the coin rest)
           ("settle_r", 0.0, 0.6),   # settle-mode translation scale (micro-motion)
           ("deadband", 0.015, 0.035))  # radius where settle mode engages (≈ center_tol)


@dataclass(frozen=True)
class CenteringParams:
    """Centering primitive parameters (see ``_RANGES`` for the searched bounds)."""

    kp: float = 18.0
    sq_hold: float = 0.8
    sq_settle: float = 0.4
    settle_r: float = 0.2
    deadband: float = 0.022

    @staticmethod
    def from_unit(u: np.ndarray) -> "CenteringParams":
        """Map a normalised ``[0,1]^5`` vector to the parameter ranges (the CEM search space)."""
        v = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        vals = [lo + v[i] * (hi - lo) for i, (_n, lo, hi) in enumerate(_RANGES)]
        return CenteringParams(*vals)


def centering_action(inner, params: CenteringParams) -> np.ndarray:
    """The centering/settling control law (6-DoF cooperative action).

    Far from the centre: translate the grasp midpoint toward the zone centre with magnitude ``clip(kp·d, 0, 1)`` (P
    control → decelerate on approach), holding the grasp (``sq_hold``). Inside the ``deadband``: settle — tiny
    proportional micro-motion (``settle_r``) and a gentler squeeze (``sq_settle``) so the coin comes to rest.

    # Preconditions the coin is (about to be) grasped; ``inner`` exposes ``_planar_metrics`` + zone. """
    d, n = _dir_to_zone(inner)                                  # unit direction to zone centre, n = disk_to_zone
    if n <= params.deadband:                                    # settle mode near the centre
        trans = params.settle_r * min(1.0, params.kp * n) * d
        sq = params.sq_settle
    else:
        trans = min(1.0, params.kp * n) * d                    # proportional: far → full, near → small (no overshoot)
        sq = params.sq_hold
    return np.array([trans[0], trans[1], 0.0, sq, 0.0, 0.0], dtype=np.float32)


def roll_centering(env: CoinDeliveryTrainEnv, seed: int, params: CenteringParams) -> dict:
    """Roll the centering primitive on one state (grasp_carry prefix → centering suffix, zero residual)."""
    env._base_override = lambda inner, _t: centering_action(inner, params)
    try:
        return roll_delivery(env, seed, scripted_action_fn())  # raw=0 → a_exec = clip(centering_action)
    finally:
        env._base_override = None


def eval_centering(params: CenteringParams, seeds, cfg: DeliveryRLConfig | None = None,
                   *, env: CoinDeliveryTrainEnv | None = None) -> dict:
    """Aggregate centering metrics over ``seeds`` (center-reach / zone-entry / final dtz / times / contact-loss)."""
    cfg = cfg or DeliveryRLConfig()
    env = env or make_delivery_rl_env(cfg)
    rows = [roll_centering(env, s, params) for s in seeds]
    n = max(1, len(rows))

    def _rate(key: str) -> float:
        return round(sum(bool(r[key]) for r in rows) / n, 4)

    def _med(key: str, subset: list | None = None) -> "float | None":
        vals = [r[key] for r in (subset if subset is not None else rows) if r[key] is not None]
        return round(float(np.median(vals)), 4) if vals else None

    deliv = [r for r in rows if r["center_reach"]]
    return {"n": len(rows), "center_reach": _rate("center_reach"), "zone_entry": _rate("zone_entry"),
            "contact_lost": _rate("contact_lost"), "final_dtz_med": _med("final_dtz"), "min_dtz_med": _med("min_dtz"),
            "time_to_center_med": _med("time_to_center", deliv), "n_center": sum(r["center_reach"] for r in rows)}


@dataclass(frozen=True)
class CenteringOracleConfig:
    pop: int = 16
    elite: int = 5
    iters: int = 8


def cem_centering(target_seeds, cfg: DeliveryRLConfig, ocfg: CenteringOracleConfig, rng: np.random.Generator,
                  *, env: CoinDeliveryTrainEnv, log=None) -> dict:
    """CEM over the 5 centering-primitive parameters to maximise center-reach on ``target_seeds`` (a SINGLE
    generalising parameterisation, not per-state). Objective: center-reach count, tie-broken by −mean(min_dtz)."""
    dim = len(_RANGES)
    mean = np.full(dim, 0.5)
    sigma = np.full(dim, 0.3)
    best = {"score": -1e9, "params": CenteringParams(), "n_center": 0, "eval": {}}
    for it in range(ocfg.iters):
        pops = np.clip(rng.normal(mean, sigma, size=(ocfg.pop, dim)), 0.0, 1.0)
        if it == 0:
            pops[0] = 0.5                                       # include the default parameterisation
        scored = []
        for u in pops:
            p = CenteringParams.from_unit(u)
            ev = eval_centering(p, target_seeds, cfg, env=env)
            score = ev["n_center"] - 0.5 * (ev["min_dtz_med"] or 1.0)   # count, tie-broken by closeness
            scored.append((score, u, p, ev))
        scored.sort(key=lambda x: x[0])
        elites = np.stack([u for _s, u, _p, _e in scored[-ocfg.elite:]])
        mean, sigma = elites.mean(0), elites.std(0) + 1e-2
        top = scored[-1]
        if top[0] > best["score"]:
            best = {"score": top[0], "params": top[2], "n_center": top[3]["n_center"], "eval": top[3]}
        if log is not None:
            log(f"  [cem-center it {it + 1}/{ocfg.iters}] best n_center={best['n_center']}/{len(list(target_seeds))} "
                f"kp={best['params'].kp:.1f} db={best['params'].deadband:.3f}")
    return best
