"""R11.6D Phase 4.1 — handoff-conditioned transport predictor.

The per-theta signature class is capped at the descriptor-nearest baseline because transport is (theta x handoff)-
dependent: a theta transports differently from different handoffs, so a per-theta scalar cannot say whether it delivers
from THIS handoff. This predicts delivered dtz for a (theta, handoff) PAIR from transport-interaction features (the
theta's transport profile crossed with the handoff's required transport + bearing), fit closed-form (ridge) on the
train transfer matrix, and selects the top-1 theta by minimum predicted dtz. Interpretable (linear coefficients),
train-only, top-1, no blending.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.coin_delivery.transport_retrieval import TransportSignature, _angle_gap

FEATURE_NAMES = ("typical", "p90", "iqr", "k6_rate", "contact_rate", "d_req", "bearing",
                 "d_req_minus_typical", "d_req_minus_p90", "angle_gap")
_DTZ_CAP_MM = 50.0               # regress a delivery-relevant capped dtz (huge misses share one bucket)


def feature_row(sig: TransportSignature, qf: "dict[str, float]") -> np.ndarray:
    """The (theta, handoff) transport-interaction feature vector (ordered as FEATURE_NAMES)."""
    d_req = qf["d_required_mm"]
    return np.array([sig.typical_transport_mm, sig.transport_p90_mm, sig.transport_iqr, sig.k6_rate, sig.contact_rate,
                     d_req, qf["bearing"], d_req - sig.typical_transport_mm, d_req - sig.transport_p90_mm,
                     _angle_gap(qf["bearing"], sig.angle_lo, sig.angle_hi)], np.float64)


@dataclass(frozen=True)
class RidgePredictor:
    """A standardized closed-form ridge predictor of capped delivered dtz for a (theta, handoff) pair."""

    w: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(phi: np.ndarray, y: np.ndarray, lam: float = 10.0) -> "RidgePredictor":
        phi = np.atleast_2d(np.asarray(phi, np.float64))
        mean, std = phi.mean(0), phi.std(0)
        std[std < 1e-9] = 1.0
        z = (phi - mean) / std
        a = np.hstack([z, np.ones((z.shape[0], 1))])
        reg = lam * np.eye(a.shape[1])
        reg[-1, -1] = 0.0                                            # do not regularize the bias
        wf = np.linalg.solve(a.T @ a + reg, a.T @ np.asarray(y, np.float64))
        return RidgePredictor(wf[:-1], float(wf[-1]), mean, std)

    def predict(self, phi: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(np.asarray(phi, np.float64)) - self.mean) / self.std
        return z @ self.w + self.bias

    def coefficients(self) -> "dict[str, float]":
        return {n: round(float(c), 3) for n, c in zip(FEATURE_NAMES, self.w)}


def capped_dtz(dtz_mm: float) -> float:
    return min(float(dtz_mm), _DTZ_CAP_MM)


def training_rows(cells: "list[dict]", sigs: "dict[str, TransportSignature]", qf: "dict[str, float]",
                  drop_handoffs: "frozenset[str]" = frozenset(), drop_theta: "str | None" = None
                  ) -> "tuple[np.ndarray, np.ndarray]":
    """(feature matrix, capped-dtz labels) from the TRAIN (theta, handoff) cells, dropping a leave-one-scenario-out
    handoff row and/or theta column."""
    phi, y = [], []
    for c in cells:
        if (c.get("split") == "train" and "error" not in c and c["handoff"] not in drop_handoffs
                and c["theta"] != drop_theta and c["theta"] in sigs and c["handoff"] in qf):
            phi.append(feature_row(sigs[c["theta"]], qf[c["handoff"]]))
            y.append(capped_dtz(c["dtz_mm"]))
    return np.array(phi, np.float64), np.array(y, np.float64)


def select_top1(pred: RidgePredictor, qf: "dict[str, float]", sigs: "dict[str, TransportSignature]",
                candidates: "list[str]") -> str:
    """The candidate theta with the smallest predicted delivered dtz from this handoff (deterministic tie-break)."""
    cand = [t for t in candidates if t in sigs]
    phi = np.array([feature_row(sigs[t], qf) for t in cand], np.float64)
    dtz = pred.predict(phi)
    return sorted(zip(dtz, cand))[0][1]
