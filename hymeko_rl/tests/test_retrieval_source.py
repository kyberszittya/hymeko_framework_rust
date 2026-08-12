"""RetrievalDeliveryPolicy.predict_with_source — the API that removes the reason call sites reimplemented the nearest
lookup: it returns the retrieved theta AND the nearest source index, and its theta equals predict()."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.retrieval import RetrievalConfig, RetrievalDeliveryPolicy, SelectRule


def _policy() -> RetrievalDeliveryPolicy:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    theta = np.array([[0.05, 0.3, 0.0, 0.0, 5.0, 0.0],
                      [0.06, 0.4, -0.1, 0.0, 6.0, 0.0],
                      [0.04, 0.2, 0.1, 0.0, 4.0, 0.0]])
    survival = np.ones(3)
    return RetrievalDeliveryPolicy.fit(x, theta, survival, RetrievalConfig(k=1, select=SelectRule.NEAREST))


def test_predict_with_source_matches_predict_and_returns_nearest() -> None:
    pol = _policy()
    q = np.array([0.9, 0.05])                        # closest to row 1
    theta, idx = pol.predict_with_source(q)
    assert idx == 1
    assert np.allclose(theta, pol.predict(q))        # predict delegates → identical


def test_predict_with_source_honours_exclude_idx() -> None:
    pol = _policy()
    q = np.array([0.0, 0.0])                          # exactly row 0
    assert pol.predict_with_source(q)[1] == 0
    assert pol.predict_with_source(q, exclude_idx=0)[1] != 0   # leave-one-out never retrieves self
