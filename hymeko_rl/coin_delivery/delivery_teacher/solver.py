"""Target-conditioned delivery+settle teacher: a per-scenario CEM over the existing ``forward_displacement.rollout_primitive``.

The primitive is *already* target-conditioned — its PUSH/BRAKE/RELEASE schedule aims at ``direction_to_zone`` (the
relocatable target) and monitors the frozen K6 certificate. This teacher re-optimizes its structured parameters for the
actual coin/target geometry, **replacing the frozen R2 downstream**. The search is driven by the existing ``score``
(K6 / distance-to-zone / contact / safety) — **energy is never in the objective** (it is measured by a non-invasive probe
on the winner only, per the R11.4A contract).

Phase A opens dimensions incrementally: it **freezes the early push** {squeeze, forward, balance} and searches only
{brake onset (``ramp``), release timing (``release``), brake gain}. Distance-to-target is controlled through the push
*duration* (``ramp``), so a fixed push magnitude/direction still reaches varied targets — which is exactly what lets a
recovery be attributed to energy-shedding rather than to a new orbit. If Phase A is insufficient the caller opens the
remaining dimensions (Phase A-2 / Phase B).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_teacher.phase_energy import PhaseEnergyLedger, PhaseEnergyProbe, build_ledger
from hymeko_rl.coin_delivery.forward_displacement import (
    ForwardConfig,
    delivery_success,
    rollout_primitive,
    score,
)

# Phase A search: the θ indices are (0 squeeze, 1 forward, 2 balance, 3 ramp, 4 release, 5 brake_gain).
PHASE_A_SEARCH = (3, 4, 5)                    # brake onset (=push duration), release timing, brake gain
PHASE_A_FROZEN = {0: 0.12, 1: 0.15, 2: 0.0}  # nominal grip + forward push + no lateral bias (early push held fixed)
FULL_SEARCH = (0, 1, 2, 3, 4, 5)             # Phase B: open the whole push (squeeze, forward magnitude, balance) too


@dataclass(frozen=True)
class DeliverySearchSpec:
    """Which θ dimensions the CEM searches (the rest are frozen), plus bounds + CEM budget. ``horizon`` is extended so
    there is room for push -> brake -> release -> settle-dwell."""

    search_idx: tuple[int, ...] = PHASE_A_SEARCH
    frozen: dict[int, float] = field(default_factory=lambda: dict(PHASE_A_FROZEN))
    lo: tuple[float, ...] = (2.0, 8.0, 0.0)       # ramp, release, brake_gain lower
    hi: tuple[float, ...] = (18.0, 30.0, 3.0)     # ramp, release, brake_gain upper
    init_std: tuple[float, ...] = (5.0, 6.0, 0.9)
    horizon: int = 36
    pop: int = 32
    iters: int = 6
    elite: int = 8

    def assemble(self, x: np.ndarray) -> np.ndarray:
        """Insert the searched values ``x`` into a full 6-vector θ, filling the frozen dimensions."""
        theta = np.zeros(6, np.float64)
        for i, v in self.frozen.items():
            theta[i] = v
        for j, idx in enumerate(self.search_idx):
            theta[idx] = float(np.clip(x[j], self.lo[j], self.hi[j]))
        return theta


@dataclass(frozen=True)
class DeliveryResult:
    """A solved delivery+settle attempt: the winning θ, the primitive measurements, the (diagnostic) energy ledger."""

    seed: int
    theta: tuple[float, ...]
    k6: bool
    safe: bool
    min_dtz_mm: float
    measurements: dict[str, Any]
    energy: PhaseEnergyLedger


def full_transport_spec(horizon: int = 90) -> DeliverySearchSpec:
    """R11.5 delivery spec: open the WHOLE push (all 6 θ) with an EXTENDED horizon, for the ~40-90 mm coin transports from
    a certified grasp to a relocated target. The shelved Phase-A default (frozen push, horizon 36) was tuned for short
    transports and cannot generalize: measured on bank_c0_0 the coin starts 85.6 mm out and is still moving toward the
    zone when a 36-step horizon ends (26.3 mm). With the full push open + horizon 90 the same state reaches strict K6."""
    return DeliverySearchSpec(
        search_idx=FULL_SEARCH, frozen={},
        lo=(0.04, 0.04, -0.12, 2.0, 8.0, 0.0), hi=(0.20, 0.45, 0.12, 30.0, 55.0, 3.5),
        init_std=(0.05, 0.10, 0.06, 6.0, 10.0, 0.9), horizon=horizon, pop=40, iters=8, elite=10)


def _config(spec: DeliverySearchSpec) -> ForwardConfig:
    return replace(ForwardConfig(), deliver=True, horizon=spec.horizon)


def _search(snap: Any, cfg: ForwardConfig, spec: DeliverySearchSpec, seed: int) -> "tuple[np.ndarray, dict[str, Any]]":
    """CEM over the searched dimensions; returns (best full-θ, best measurements). Objective = ``score`` (no energy)."""
    rng = np.random.default_rng(seed)
    dim = len(spec.search_idx)
    mean = np.array([(spec.lo[j] + spec.hi[j]) / 2.0 for j in range(dim)], np.float64)
    std = np.array(spec.init_std, np.float64)
    best: "tuple[float, np.ndarray, dict[str, Any]] | None" = None
    for _ in range(spec.iters):
        pop = mean[None] + std[None] * rng.standard_normal((spec.pop, dim))
        scored = []
        for x in pop:
            theta = spec.assemble(x)
            m = rollout_primitive(snap, theta, cfg)
            s = score(m, cfg)
            scored.append((s, x))
            if best is None or s > best[0]:
                best = (s, theta, m)
        scored.sort(key=lambda z: z[0], reverse=True)
        elite = np.stack([x for _, x in scored[:spec.elite]])
        mean, std = elite.mean(0), elite.std(0) * 0.9 + 1e-3
        if best is not None and best[2]["k6_delivered"]:
            break
    assert best is not None
    return best[1], best[2]


def solve_delivery(snap: Any, seed: int = 0, spec: DeliverySearchSpec = DeliverySearchSpec()) -> DeliveryResult:
    """Solve delivery+settle for this capture handoff toward its (relocated) target, then re-run the winner once with the
    non-invasive energy probe to build the diagnostic ledger.

    Preconditions: ``snap`` is a post-capture handoff whose ``direction_to_zone`` targets the scenario's zone.
    Postconditions: the returned θ is deterministic given ``seed``; the energy ledger is measured, not optimized.
    """
    cfg = _config(spec)
    theta, _ = _search(snap, cfg, spec, seed)
    probe = PhaseEnergyProbe(ramp=int(round(theta[3])), release=int(round(theta[4])),
                             dt=float(snap.stack.control_dt))
    m = rollout_primitive(snap, theta, cfg, frame_hook=probe)
    ledger = build_ledger(probe)
    safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
    return DeliveryResult(seed=seed, theta=tuple(round(float(t), 4) for t in theta),
                          k6=bool(delivery_success(m, cfg)), safe=safe, min_dtz_mm=round(m["dtz_end"] * 1000, 2),
                          measurements=m, energy=ledger)
