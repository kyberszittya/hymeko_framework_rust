"""R3 — the smallest-sufficient LEARNED intent predictor: canonical R1 structured state -> 7 physical-intent roles.

The R3 hypothesis is that the physical INTENT (authority-normalised) transfers where raw theta did not (R2). With only 6 dev
cradles and narrow delivery basins, the capacity-controlled baseline is a distance-weighted kernel regression
(Nadaraya-Watson) over the dev (state -> intent) pairs in normalised canonical-R1-feature space — one hyperparameter
(bandwidth), interpolative, no theta ever predicted directly. A dev cradle predicts (near) its own intent (self-weight);
a held-out cradle predicts a blend of its nearest dev intents, which the deterministic decoder then specialises to the
held-out authority. Bandwidth is selected by DEV-only leave-one-out (physical K6), never held-out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.coin_delivery.theta_option.physical_intent import INTENT_HI, INTENT_LO, PhysicalIntent

_EPS = 1e-9


@dataclass
class NWIntentPredictor:
    """Nadaraya-Watson kernel regression dev(state->intent). `predict(feat)` = Σ w_i·intent_i, w_i ∝ exp(-‖z_q-z_i‖²/2h²)
    over z-scored dev features. Deterministic. # Postconditions: predicted intent is clipped to the frozen intent box."""

    feats_z: np.ndarray                      # (N, F) z-scored dev features
    intents: np.ndarray                      # (N, 7) dev intents
    mean: np.ndarray                         # (F,) feature mean
    std: np.ndarray                          # (F,) feature std
    bandwidth: float

    def _z(self, feat: np.ndarray) -> np.ndarray:
        return (np.asarray(feat, np.float64).ravel() - self.mean) / self.std

    def predict(self, feat: np.ndarray) -> PhysicalIntent:
        zq = self._z(feat)
        d2 = ((self.feats_z - zq[None, :]) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2.0 * self.bandwidth ** 2 + _EPS))
        w = w / (w.sum() + _EPS)
        v = (w[:, None] * self.intents).sum(axis=0)
        return PhysicalIntent.from_vector(np.clip(v, INTENT_LO, INTENT_HI))


def _build(feats: np.ndarray, intents: np.ndarray, bandwidth: float) -> NWIntentPredictor:
    feats = np.asarray(feats, np.float64)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0
    return NWIntentPredictor(feats_z=(feats - mean) / std, intents=np.asarray(intents, np.float64),
                             mean=mean, std=std, bandwidth=float(bandwidth))


def fit_intent_predictor(feats: np.ndarray, intents: np.ndarray, *, bandwidth: float) -> NWIntentPredictor:
    """Fit on ALL dev (state, intent) pairs at a given bandwidth."""
    return _build(feats, intents, bandwidth)


def lodo_predictor(feats: np.ndarray, intents: np.ndarray, held: int, bandwidth: float) -> NWIntentPredictor:
    """A predictor fit on every dev cradle EXCEPT ``held`` (for dev-only leave-one-out bandwidth selection)."""
    keep = [i for i in range(len(feats)) if i != held]
    return _build(np.asarray(feats)[keep], np.asarray(intents)[keep], bandwidth)
