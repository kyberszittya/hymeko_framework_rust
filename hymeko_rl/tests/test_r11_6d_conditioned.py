"""Tests for R11.6D Phase 4.1 handoff-conditioned predictor: features, ridge fit/predict, LOSO drop, top-1 select."""
import numpy as np

from hymeko_rl.coin_delivery.transport_predictor import (
    FEATURE_NAMES,
    RidgePredictor,
    capped_dtz,
    feature_row,
    select_top1,
    training_rows,
)
from hymeko_rl.coin_delivery.transport_retrieval import TransportSignature


def _sig(typical: float, p90: float) -> TransportSignature:
    return TransportSignature(typical, p90, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0)


def test_feature_row_interaction_terms() -> None:
    f = feature_row(_sig(70.0, 100.0), {"d_required_mm": 100.0, "bearing": 0.0})
    assert len(f) == len(FEATURE_NAMES)
    assert f[7] == 30.0 and f[8] == 0.0                          # d_req-typical=30 ; d_req-p90=0 (reaches)


def test_capped_dtz() -> None:
    assert capped_dtz(12.0) == 12.0 and capped_dtz(160.0) == 50.0


def test_ridge_fits_linear_signal() -> None:
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((200, 4))
    y = 3.0 * phi[:, 0] - 2.0 * phi[:, 1] + 5.0                   # exact linear
    p = RidgePredictor.fit(phi, y, lam=1e-6)
    assert np.allclose(p.predict(phi), y, atol=0.2)


def _cells() -> list:
    # theta tFar reaches (dtz 5) the far handoff hFar (d_req 100); tNear undershoots it (dtz 40); vice versa on hNear.
    return [{"handoff": "hFar", "theta": "tFar", "split": "train", "dtz_mm": 5.0, "k6": True, "safe": True},
            {"handoff": "hFar", "theta": "tNear", "split": "train", "dtz_mm": 40.0, "k6": False, "safe": True},
            {"handoff": "hNear", "theta": "tFar", "split": "train", "dtz_mm": 30.0, "k6": False, "safe": True},
            {"handoff": "hNear", "theta": "tNear", "split": "train", "dtz_mm": 6.0, "k6": True, "safe": True}]


def test_training_rows_drops_loso_handoff_and_theta() -> None:
    sigs = {"tFar": _sig(90.0, 110.0), "tNear": _sig(60.0, 70.0)}
    qf = {"hFar": {"d_required_mm": 100.0, "bearing": 0.0}, "hNear": {"d_required_mm": 60.0, "bearing": 0.0}}
    phi, y = training_rows(_cells(), sigs, qf, drop_handoffs=frozenset({"hFar"}), drop_theta="tNear")
    assert phi.shape[0] == 1 and y[0] == 30.0                    # only (hNear, tFar) survives the drops


def test_select_top1_picks_min_predicted_dtz() -> None:
    sigs = {"tFar": _sig(90.0, 110.0), "tNear": _sig(60.0, 70.0)}
    qf = {"hFar": {"d_required_mm": 100.0, "bearing": 0.0}, "hNear": {"d_required_mm": 60.0, "bearing": 0.0}}
    phi, y = training_rows(_cells(), sigs, qf)
    pred = RidgePredictor.fit(phi, y, lam=0.01)
    # for the far handoff, the far-reaching theta should be selected (lower predicted dtz)
    assert select_top1(pred, qf["hFar"], sigs, ["tFar", "tNear"]) == "tFar"
