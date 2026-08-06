"""COIN-DELIVERY-RL-2 — hard-state classification, residual ORACLE, and the HyMeKo hard-state problem generator.

COIN-DELIVERY-RL-1 was CASE D: a uniform PPO residual matched but did not beat scripted grasp_carry (residual ~0). The
open question is whether the remaining HARD (scripted-failure) states are recoverable WITHIN the residual/action
abstraction. This module answers it WITHOUT RL:

  * ``classify_held`` — sort the 90 held states into 7 failure classes from the scripted rollout metrics + start
    geometry (Stage 0).
  * ``oracle_recoverability`` — a bounded open-loop optimizer (segmented CEM over a residual SEQUENCE) sweeping residual
    scale δ∈{0.30,0.50,0.75,1.00} on ONLY the failures, labelling each RECOVERABLE_CURRENT_ABSTRACTION /
    RECOVERABLE_WITH_WIDER_RESIDUAL / REQUIRES_NEW_PRIMITIVE / APPARENTLY_UNREACHABLE (Stage 1). At δ=1.0 the residual
    can override grasp_carry entirely, so δ=1.0 ≈ near-arbitrary open-loop control — a strong upper bound on what a
    bounded residual could do.
  * ``CoinDeliveryProblem`` + ``CoinDeliveryProblemGenerator`` — a generator that oversamples RECOVERABLE hard states
    for targeted training (Stage 2), only used if the Stage-1 gate passes.

Reuses ``train.coin_delivery_rl`` (the harness); NO RL here, NO env/reward/dynamics/CORE change. The oracle is a
measurement of the recoverability CEILING, the gate for whether targeted RL (Stage 3) is even justified.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hymeko_rl.train.coin_delivery_rl import CoinDeliveryTrainEnv, DeliveryRLConfig, roll_delivery, scripted_action_fn


class FailureClass(str, Enum):
    CENTER_SUCCESS = "SCRIPTED_CENTER_SUCCESS"
    ZONE_ONLY = "ZONE_ONLY"
    NEAR_MISS = "NEAR_MISS"
    CONTACT_LOSS = "CONTACT_LOSS"
    TRANSPORT_STALL = "TRANSPORT_STALL"
    GEOMETRIC_HARD = "GEOMETRIC_HARD"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class Recoverability(str, Enum):
    CURRENT = "RECOVERABLE_CURRENT_ABSTRACTION"
    WIDER = "RECOVERABLE_WITH_WIDER_RESIDUAL"
    NEW_PRIMITIVE = "REQUIRES_NEW_PRIMITIVE"
    UNREACHABLE = "APPARENTLY_UNREACHABLE"


@dataclass(frozen=True)
class CoinDeliveryProblem:
    """A classified hard-state problem (HyMeKo hard-state generator unit)."""

    state_id: str
    failure_class: str
    recoverability: str
    start_disk_to_zone: float
    required_residual_scale: float          # the smallest δ that recovered it (NaN if never recovered)
    monitor_targets: tuple[str, ...]        # which delivery monitors this state stresses (e.g. "center", "contact")


# ── Stage 0 — classification ─────────────────────────────────────────────────────────────────────────────────────────
def classify_state(row: dict, *, zone_half: float, near_band: float = 1.5,
                   stall_abs: float = 0.02, stall_frac: float = 0.2) -> FailureClass:
    """Classify one scripted rollout into a failure class (precedence: success → zone → near → contact → stall → geom).

    # Preconditions ``row`` has the ``roll_delivery`` keys; ``zone_half > 0``.
    # Postconditions returns exactly one :class:`FailureClass`."""
    if row["center_reach"]:
        return FailureClass.CENTER_SUCCESS
    if row["zone_entry"]:
        return FailureClass.ZONE_ONLY
    min_dtz = float(row["min_dtz"])
    if min_dtz <= zone_half * near_band:                    # came within ~1.5·zone_half of the boundary
        return FailureClass.NEAR_MISS
    if row["contact_lost"]:                                 # grasped then dropped, never near the zone
        return FailureClass.CONTACT_LOSS
    progress = float(row["start_dtz"]) - min_dtz
    if row["handoff_event"] and progress < max(stall_abs, stall_frac * float(row["start_dtz"])):
        return FailureClass.TRANSPORT_STALL                # grasped but did not transport
    if not row["handoff_event"]:
        return FailureClass.GEOMETRIC_HARD                 # never established a stable grasp
    return FailureClass.UNKNOWN_FAILURE


def classify_held(cfg: DeliveryRLConfig, held, *, env: CoinDeliveryTrainEnv | None = None) -> list[dict]:
    """Classify every held state under the scripted controller. Returns per-state {seed, class, row, geom}."""
    if env is None:
        from hymeko_rl.train.coin_delivery_rl import make_delivery_rl_env
        env = make_delivery_rl_env(cfg)
    af = scripted_action_fn()
    out = []
    for sd in held:
        row = roll_delivery(env, sd, af)
        geom = env.start_geometry()
        out.append({"seed": int(sd), "failure_class": classify_state(row, zone_half=cfg.zone_half).value,
                    "row": row, "geom": geom})
    return out


# ── Stage 1 — residual oracle (segmented open-loop CEM) ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OracleConfig:
    segments: int = 4
    pop: int = 16
    elite: int = 5
    iters: int = 6
    horizon: int = 80


def _roll_residual_seq(env: CoinDeliveryTrainEnv, seed: int, theta: np.ndarray, delta: float,
                       horizon: int, segments: int) -> dict:
    """Roll a segmented open-loop residual sequence ``theta`` (segments×6) at scale ``delta``; capture delivery."""
    env._delta_override = float(delta)
    _obs, _info = env.reset(seed=int(seed))
    start_dtz = env._start_dtz
    seg_len = max(1, horizon // segments)
    min_dtz = start_dtz
    center = zone = False
    t_center = None
    both_steps = 0
    t = 0
    for t in range(horizon):
        raw = theta[min(t // seg_len, segments - 1)]
        _obs, _r, term, trunc, info = env.step(raw)
        dtz = float(info["disk_to_zone"])
        min_dtz = min(min_dtz, dtz)
        center = center or bool(info["center_reached"])
        zone = zone or bool(info["delivery_success"])
        both_steps += int(env._both())
        if info["center_reached"] and t_center is None:
            t_center = t
        if term or trunc:
            break
    return {"center": center, "zone": zone, "min_dtz": min_dtz, "start_dtz": start_dtz,
            "t_center": t_center, "contact_frac": both_steps / max(1, t + 1)}


def _score(m: dict) -> float:
    """Lexicographic delivery objective: center >> zone >> progress >> contact preservation."""
    return (100.0 * float(m["center"]) + 10.0 * float(m["zone"])
            + (m["start_dtz"] - m["min_dtz"]) + 0.05 * m["contact_frac"])


def residual_oracle_cem(env: CoinDeliveryTrainEnv, seed: int, delta: float, ocfg: OracleConfig,
                        rng: np.random.Generator) -> dict:
    """CEM over a segmented open-loop residual sequence at scale ``delta``. Returns the best rollout's metrics."""
    dim = ocfg.segments * 6
    mean = np.zeros(dim, np.float32)
    sigma = np.ones(dim, np.float32)
    best: dict = {"center": False, "zone": False, "min_dtz": np.inf, "start_dtz": 0.0, "t_center": None,
                  "contact_frac": 0.0, "score": -1e9}
    for it in range(ocfg.iters):
        pops = rng.normal(mean, sigma, size=(ocfg.pop, dim)).astype(np.float32)
        if it == 0:
            pops[0] = 0.0                                   # include the zero residual (scripted) as a candidate
        scored = []
        for p in pops:
            m = _roll_residual_seq(env, seed, p.reshape(ocfg.segments, 6), delta, ocfg.horizon, ocfg.segments)
            m["score"] = _score(m)
            scored.append(m)
        order = np.argsort([m["score"] for m in scored])
        elites = pops[order[-ocfg.elite:]]
        mean, sigma = elites.mean(0), elites.std(0) + 1e-2
        top = scored[int(order[-1])]
        if top["score"] > best["score"]:
            best = top
    return best


def oracle_recoverability(env: CoinDeliveryTrainEnv, seed: int, deltas: tuple[float, ...], ocfg: OracleConfig,
                          rng: np.random.Generator, *, zone_half: float) -> dict:
    """Adaptive ascending-δ sweep on one failure: stop at the smallest δ that recovers center; label recoverability."""
    per_delta: dict[float, dict] = {}
    recovered_at: float | None = None
    for d in sorted(deltas):
        best = residual_oracle_cem(env, seed, d, ocfg, rng)
        per_delta[d] = {"center": best["center"], "zone": best["zone"], "min_dtz": round(best["min_dtz"], 4),
                        "t_center": best["t_center"]}
        if best["center"]:
            recovered_at = d
            break
    if recovered_at is not None and recovered_at <= min(deltas):
        rec = Recoverability.CURRENT
    elif recovered_at is not None:
        rec = Recoverability.WIDER
    else:                                                   # never centred at any δ — how close did the widest get?
        widest = per_delta[max(per_delta)]
        rec = Recoverability.NEW_PRIMITIVE if (widest["zone"] or widest["min_dtz"] <= zone_half * 1.5) \
            else Recoverability.UNREACHABLE
    min_h = per_delta.get(recovered_at, {}).get("t_center") if recovered_at is not None else None
    return {"recoverability": rec.value, "recovered_at_delta": recovered_at, "min_H": min_h, "per_delta": per_delta}


def _monitor_targets(failure_class: str) -> tuple[str, ...]:
    """Which delivery monitors a failure class stresses (for the problem's monitor_targets)."""
    return {FailureClass.ZONE_ONLY.value: ("center",), FailureClass.NEAR_MISS.value: ("center", "zone"),
            FailureClass.CONTACT_LOSS.value: ("contact", "center"), FailureClass.TRANSPORT_STALL.value: ("transport",),
            FailureClass.GEOMETRIC_HARD.value: ("acquisition", "transport")}.get(failure_class, ("center",))


def build_problems(classified: list[dict], oracle: dict[int, dict]) -> list[CoinDeliveryProblem]:
    """Fuse Stage-0 classes with Stage-1 recoverability into ``CoinDeliveryProblem`` records (successes marked CURRENT)."""
    problems = []
    for c in classified:
        sd = c["seed"]
        if c["failure_class"] == FailureClass.CENTER_SUCCESS.value:
            rec, scale = Recoverability.CURRENT.value, 0.0
        else:
            orc = oracle.get(sd, {})
            rec = orc.get("recoverability", Recoverability.UNREACHABLE.value)
            scale = orc.get("recovered_at_delta") or float("nan")
        problems.append(CoinDeliveryProblem(
            state_id=str(sd), failure_class=c["failure_class"], recoverability=rec,
            start_disk_to_zone=round(float(c["row"]["start_dtz"]), 4), required_residual_scale=float(scale),
            monitor_targets=_monitor_targets(c["failure_class"])))
    return problems


# ── Stage 2 — hard-state problem generator ───────────────────────────────────────────────────────────────────────────
class CoinDeliveryProblemGenerator:
    """Samples held seeds weighted toward RECOVERABLE hard states (oversampling), with a small easy-state mix so the
    targeted policy does not forget the already-solved states. Impossible states get zero weight."""

    _RECOVERABLE = frozenset({Recoverability.CURRENT.value, Recoverability.WIDER.value})

    def __init__(self, problems: list[CoinDeliveryProblem], *, easy_weight: float = 0.2) -> None:
        self.problems = problems
        weights = []
        for p in problems:
            if p.failure_class == FailureClass.CENTER_SUCCESS.value:
                weights.append(easy_weight)                # keep a few easy states in the mix (preservation)
            elif p.recoverability in self._RECOVERABLE:
                weights.append(1.0)                        # the targeting mass
            else:
                weights.append(0.0)                        # REQUIRES_NEW_PRIMITIVE / UNREACHABLE — not trainable here
        total = float(sum(weights)) or 1.0
        self._weights = np.asarray([w / total for w in weights], dtype=np.float64)
        self._seeds = np.asarray([int(p.state_id) for p in problems])

    @property
    def recoverable_seeds(self) -> list[int]:
        """Recoverable HARD states (the Stage-3 training targets) — excludes already-successful states."""
        return [int(p.state_id) for p in self.problems
                if p.recoverability in self._RECOVERABLE and p.failure_class != FailureClass.CENTER_SUCCESS.value]

    def delta_for(self, seed: int, default: float) -> float:
        """Per-class residual scale = the state's required_residual_scale (else ``default``)."""
        for p in self.problems:
            if int(p.state_id) == int(seed) and np.isfinite(p.required_residual_scale) and p.required_residual_scale > 0:
                return float(p.required_residual_scale)
        return float(default)

    def sample_seed(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self._seeds, p=self._weights))
