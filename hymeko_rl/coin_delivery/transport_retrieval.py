"""R11.6D Phase 4 — transportability-aware retrieval.

The R11.6D audit showed the c3 far-angle failures are a TRANSPORT MISMATCH: the retrieved theta transports the right
amount for the nearer neighbour target but undershoots the farther one, and no handoff-state component is the cause. This
module replaces descriptor-nearest retrieval with a small, physically-interpretable transportability RANKER: it matches
each theta's empirical transport signature (from a full train theta x handoff transfer matrix) to the query's REQUIRED
transport, and selects ONE complete stored theta (no blending, no runtime oracle/CEM).

Score (weights fit train-only, undershoot alpha and overshoot beta SEPARATE so the matrix decides which dominates):
    S(s, theta_i) = -alpha*max(0, d_req - d_hat_i) - beta*max(0, d_hat_i - d_req)
                    - gamma*angle_gap_i + eta*k6_rate_i + rho*contact_rate_i.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive

_UNDERSHOOT_TOL_MM = 5.0          # a cell counts as under/overshoot beyond this projected-transport error


def roll_full(snap: Any, theta: np.ndarray) -> "dict[str, Any]":
    """One pure delivery rollout with the RICH transport metrics for a transfer-matrix cell."""
    m = rollout_primitive(snap, clip_theta(np.asarray(theta, np.float64)), CLOSED_LOOP_CFG)
    n = max(1, len(m["coin_trace"]))
    dtz0, fwd = float(m["dtz_start"]) * 1000, float(m["forward"]) * 1000
    traj_hash = hashlib.md5(np.asarray(m["coin_trace"], np.float64).round(4).tobytes()).hexdigest()[:12]  # noqa: S324
    return {"k6": bool(delivery_success(m, CLOSED_LOOP_CFG)),
            "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5),
            "dtz_mm": round(float(m["dtz_end"]) * 1000, 2), "d_required_mm": round(dtz0, 2),
            "projected_transport_mm": round(fwd, 2), "lateral_mm": round(abs(float(m["cross"])) * 1000, 2),
            "undershoot_mm": round(max(0.0, dtz0 - fwd), 2), "overshoot_mm": round(max(0.0, fwd - dtz0), 2),
            "contact_retention": round(1.0 - int(m["contact_lost_steps"]) / n, 3),
            "entry_speed": round(float(m["terminal_coin_speed"]), 4), "gap_closed": round(float(m["gap_closed"]), 3),
            "traj_hash": traj_hash}


def query_features(snap: Any) -> "dict[str, float]":
    """The required-transport features at a handoff: distance to the zone and the target bearing (coin->zone angle)."""
    rl = snap.branch()
    u, dtz = rl.inner.direction_to_zone()
    u = np.asarray(u, np.float64)
    return {"d_required_mm": round(float(dtz) * 1000, 2), "bearing": round(float(np.arctan2(u[1], u[0])), 5)}


@dataclass(frozen=True)
class TransportSignature:
    """One theta's empirical transport behaviour, summarised over the train handoffs it was rolled from."""

    typical_transport_mm: float
    transport_p90_mm: float       # a theta's REACH: the 90th pct of projected transport (can it reach far?)
    transport_iqr: float
    angle_lo: float
    angle_hi: float
    undershoot_freq: float
    overshoot_freq: float
    k6_rate: float
    contact_rate: float


def signature_from_cells(cells: "list[dict]") -> TransportSignature:
    """Build a theta's signature from its transfer-matrix column (one cell per source handoff, carrying that handoff's
    ``bearing`` and the rollout metrics). ``typical_transport`` is the median projected transport; ``transport_p90`` its
    reach; the supported angle range spans the bearings where the theta delivered strict-K6 (else all, as a weak prior)."""
    tr = np.array([c["projected_transport_mm"] for c in cells], np.float64)
    k6 = np.array([c["k6"] for c in cells], bool)
    bearings = np.array([c["bearing"] for c in cells], np.float64)
    hit_b = bearings[k6] if k6.any() else bearings
    return TransportSignature(
        typical_transport_mm=round(float(np.median(tr)), 2), transport_p90_mm=round(float(np.percentile(tr, 90)), 2),
        transport_iqr=round(float(np.subtract(*np.percentile(tr, [75, 25]))), 2),
        angle_lo=round(float(hit_b.min()), 5), angle_hi=round(float(hit_b.max()), 5),
        undershoot_freq=round(float(np.mean([c["undershoot_mm"] > _UNDERSHOOT_TOL_MM for c in cells])), 3),
        overshoot_freq=round(float(np.mean([c["overshoot_mm"] > _UNDERSHOOT_TOL_MM for c in cells])), 3),
        k6_rate=round(float(k6.mean()), 3), contact_rate=round(float(np.mean([c["contact_retention"] for c in cells])), 3))


@dataclass(frozen=True)
class TransportWeights:
    alpha: float = 1.0            # undershoot penalty (d_req > transport)
    beta: float = 1.0            # overshoot penalty (transport > d_req)
    gamma: float = 0.0           # target-angle mismatch (mm-equivalent per rad)
    eta: float = 0.0             # empirical K6 rate (mm-equivalent)
    rho: float = 0.0             # contact-retention rate (mm-equivalent)


def _angle_gap(bearing: float, lo: float, hi: float) -> float:
    return max(0.0, lo - bearing, bearing - hi)


def score(qf: "dict[str, float]", sig: TransportSignature, w: TransportWeights, reach: bool = False) -> float:
    """The transportability score (higher is better). Undershoot and overshoot are penalised separately. ``reach``:
    judge undershoot against the theta's REACH (90th-pct transport) rather than its median, so a theta that CAN transport
    far (but typically doesn't) is not deprioritised for a far target (overshoot stays judged by the median)."""
    d_req = qf["d_required_mm"]
    reach_mm = sig.transport_p90_mm if reach else sig.typical_transport_mm
    under, over = max(0.0, d_req - reach_mm), max(0.0, sig.typical_transport_mm - d_req)
    return (-w.alpha * under - w.beta * over - w.gamma * _angle_gap(qf["bearing"], sig.angle_lo, sig.angle_hi)
            + w.eta * sig.k6_rate + w.rho * sig.contact_rate)


def rank_theta(qf: "dict[str, float]", sigs: "dict[str, TransportSignature]", w: TransportWeights,
               reach: bool = False) -> "list[str]":
    """All theta-ids ranked best-first by the transportability score (deterministic tie-break on id)."""
    return sorted(sigs, key=lambda tid: (score(qf, sigs[tid], w, reach), tid), reverse=True)


# a small pre-registered weight grid (physical score); MLP only if train-only CV shows a clear nonlinear gap.
WEIGHT_GRID: "tuple[TransportWeights, ...]" = tuple(
    TransportWeights(alpha=a, beta=b, gamma=g, eta=e, rho=r)
    for a in (1.0, 2.0, 4.0) for b in (0.5, 1.0) for g in (0.0, 20.0, 50.0)
    for e in (0.0, 30.0) for r in (0.0, 20.0))


def build_signatures(cells: "list[dict]", exclude: "frozenset[str]" = frozenset()) -> "dict[str, TransportSignature]":
    """Per-theta transport signatures from the TRAIN-handoff cells (dev never enters a signature), optionally excluding
    a set of handoffs (the leave-one-scenario-out query's own demonstration)."""
    by_theta: "dict[str, list]" = defaultdict(list)
    for c in cells:
        if c.get("split") == "train" and c["handoff"] not in exclude and "error" not in c:
            by_theta[c["theta"]].append(c)
    return {tid: signature_from_cells(cs) for tid, cs in by_theta.items()}


def cell_index(cells: "list[dict]") -> "dict[tuple[str, str], dict]":
    """(handoff, theta) -> cell, for O(1) matrix lookup."""
    return {(c["handoff"], c["theta"]): c for c in cells if "error" not in c}


def _delivers(idx: "dict[tuple[str, str], dict]", handoff: str, tid: str) -> bool:
    c = idx.get((handoff, tid))
    return bool(c and c["k6"] and c["safe"])


def evaluate_handoff(idx: "dict[tuple[str, str], dict]", handoff: str, qf: "dict[str, float]",
                     sigs: "dict[str, TransportSignature]", candidates: "list[str]", w: TransportWeights,
                     k: int = 3, reach: bool = False) -> "dict[str, Any]":
    """Top-1 transportability selection at one handoff: which theta is chosen, whether it delivers, top-k coverage, and
    oracle regret (selected dtz - best-available deliverable dtz among the candidates)."""
    ranked = rank_theta(qf, {t: sigs[t] for t in candidates if t in sigs}, w, reach)
    top1 = ranked[0]
    sel = idx.get((handoff, top1))
    deliver_dtz = [idx[(handoff, t)]["dtz_mm"] for t in candidates if _delivers(idx, handoff, t)]
    oracle = min(deliver_dtz) if deliver_dtz else None
    regret = round(sel["dtz_mm"] - oracle, 2) if (sel and oracle is not None) else None
    return {"top1": top1, "k6": _delivers(idx, handoff, top1), "sel_dtz": sel["dtz_mm"] if sel else None,
            "top3_deliverable": any(_delivers(idx, handoff, t) for t in ranked[:k]),
            "oracle_dtz": oracle, "regret": regret, "n_deliverable": len(deliver_dtz)}
