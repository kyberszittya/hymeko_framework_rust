"""STAGE 0/5 (shared) — the FIXED bounded structured search over the 6-D torque-θ, and the single source of truth that
both the update-0 deploy and the RL environment call.

The learned actor proposes a centre θ_0; this search samples ``budget`` candidates around θ_0 (frozen Gaussian jitter in
normalised z-space), rolls each through the FROZEN physical option (`rollout_primitive`, no pin / teleport / coin edit),
and keeps the argmax of the frozen canonical delivery `score`. The selected θ_exec is returned as *provenance*; the
Bellman action stays θ_0. The generator/scorer/budget are FROZEN across a run (a stationary environment response — the
Bellman-safety contract of `option_rl.FixedBudgetSearch`, which this reuses).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.forward_displacement import (
    ForwardConfig, delivery_success, rollout_primitive, score as fd_score)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox, ThetaProvenance

# FROZEN per-component search spread in normalised z-space (uniform across the 6 components; ≈0.15 matches the carry
# adapter's std_amp/std_dur once normalised). Bounded, local — the search refines the proposal, it does not re-solve.
SEARCH_STD = 0.15


@dataclass
class ThetaCandidateGenerator:
    """`option_rl.CandidateGenerator`: sample ``n`` legal-θ candidates around a legal-θ centre by jittering in NORMALISED
    z-space with a frozen std, then denorm-clipping to the box (so every candidate is a legal option regardless of the
    box asymmetry). Deterministic given the rng. # Postconditions: returns (n, 6) legal θ."""

    std: float = SEARCH_STD
    box: ThetaBox = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.box = self.box or ThetaBox()

    def sample(self, center: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        z0 = self.box.norm(center)
        z = z0[None, :] + self.std * rng.standard_normal((int(n), z0.shape[0]))
        return np.asarray([self.box.denorm(zi) for zi in z], np.float32)


def _compact_outcome(m: dict[str, Any], cfg: ForwardConfig) -> dict[str, Any]:
    """The audit-compact outcome kept in provenance (never the label): the frozen K6 verdict + the physical margins."""
    return {"k6_delivered": bool(m["k6_delivered"]), "k6_max_dwell": int(m["k6_max_dwell"]),
            "dtz_end": float(m["dtz_end"]), "dtz_start": float(m["dtz_start"]),
            "terminal_coin_speed": float(m["terminal_coin_speed"]), "peak_qdot": float(m["peak_qdot"]),
            "peak_coin_speed": float(m["peak_coin_speed"]), "forward": float(m["forward"]),
            "touched": bool(m["touched"]), "delivery_success": bool(delivery_success(m, cfg))}


@dataclass
class ThetaCandidateScorer:
    """`option_rl.CandidateScorer` bound to ONE frozen `CradleSnapshot`. `score(candidate, rng)` rolls the candidate θ
    through the frozen physical option and returns (frozen delivery score, compact outcome). PURE physics — no pin, no
    teleport, no coin-state edit (all inside `rollout_primitive`). # Preconditions: candidate is a legal θ (length 6)."""

    snap: CradleSnapshot
    cfg: ForwardConfig = DELIVERY_CFG

    def score(self, candidate: np.ndarray, rng: np.random.Generator) -> tuple[float, dict[str, Any]]:
        m = rollout_primitive(self.snap, tuple(np.asarray(candidate, np.float64)), self.cfg)
        return float(fd_score(m, self.cfg)), _compact_outcome(m, self.cfg)


def search_semantics(budget: int) -> dict[str, Any]:
    """The FROZEN, explicit budget accounting for the centre-inclusive fixed search. ``budget`` is the TOTAL number of
    candidate evaluations, of which the centre θ_0 is always one and ``budget-1`` are jittered neighbours (``budget≤1`` ⇒
    the centre alone). Recorded in the artifacts so the compute budget is unambiguous and identical across conditions."""
    b = int(budget)
    return {"budget_counts_total_candidates": True, "centre_always_evaluated": True,
            "n_total_candidates": max(1, b), "n_jittered_candidates": max(0, b - 1),
            "budget_zero": "direct execution of θ_0 (centre only, no jitter)", "search_std_norm": SEARCH_STD}


def fixed_search_select(snap: CradleSnapshot, center: np.ndarray, rng: np.random.Generator, *,
                        budget: int, cfg: ForwardConfig = DELIVERY_CFG, std: float = SEARCH_STD,
                        scorer: "ThetaCandidateScorer | None" = None) -> ThetaProvenance:
    """The single deploy/RL search wrapper (CENTRE-INCLUSIVE local search). Evaluate the centre θ_0 AND ``budget-1``
    jittered neighbours — ``budget`` TOTAL evaluations, the centre always among them — keep the argmax of the frozen
    delivery score, and return a `ThetaProvenance` separating θ_0 (Bellman action) from θ_exec (selected). ``budget≤1``
    executes θ_0 directly (no jitter).

    Centre-inclusion is a deliberate correctness fix over `option_rl.FixedBudgetSearch` (which scores only the jittered
    neighbours): a search that never evaluates its own centre can DISCARD a delivering θ_0 for a worse neighbour and is
    non-monotone in budget. This bit the coin's narrow delivering basins (a jittered candidate escaped the basin). The
    change is uniform across informed/uninformed/oracle conditions — it is NOT enlarging the search for the learned actor,
    and NOT held-out tuning. Reuses the engine's `CandidateGenerator`/`CandidateScorer` protocols; ``scorer`` may be
    injected (tests). Deterministic given ``rng``. # Postconditions: `prov.center` is the input θ_0 (clipped legal);
    `prov.selected` is a legal θ; both are separate objects; exactly ``max(1,budget)`` candidates are evaluated; the
    selected score ≥ the centre's score (monotone in budget)."""
    box = ThetaBox()
    center = box.clip(center)
    scorer = scorer or ThetaCandidateScorer(snap=snap, cfg=cfg)
    cands = [center]
    if int(budget) > 1:
        gen = ThetaCandidateGenerator(std=std, box=box)
        cands += list(gen.sample(center, int(budget) - 1, rng))
    best_i, best_sc, best_out = 0, -np.inf, None
    for i, c in enumerate(cands):
        sc, out = scorer.score(np.asarray(c, np.float64), rng)
        if best_out is None or sc > best_sc:
            best_i, best_sc, best_out = i, sc, out
    return ThetaProvenance(center=np.asarray(center, np.float32), selected=np.asarray(cands[best_i], np.float32),
                           score=float(best_sc), budget=len(cands), outcome=best_out or {})
