"""Tests for the R11.5R retrieval delivery policy: parity with the baseline, the select rules, LOO, box-clip, cert."""
import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.models import NearestSchedulePolicy, THETA_HI, THETA_LO
from hymeko_rl.coin_delivery.delivery_bc.retrieval import (
    RetrievalConfig,
    RetrievalDeliveryPolicy,
    RetrievalDeploymentCertificate,
    SelectRule,
)

_MID = (THETA_LO + THETA_HI) / 2.0


def _data(n: int = 6, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    T = THETA_LO + rng.random((n, 6)) * (THETA_HI - THETA_LO)     # in-box thetas
    surv = rng.random(n)
    return X, T, surv


def test_parity_with_nearest_schedule_policy() -> None:
    X, T, surv = _data()
    ret = RetrievalDeliveryPolicy.fit(X, T, surv, RetrievalConfig(standardize=True, k=1, select=SelectRule.NEAREST))
    base = NearestSchedulePolicy.fit(X, T)
    for q in (X[0], X[3] + 0.1, np.zeros(5)):
        assert np.allclose(ret.predict(q), base.predict(q))       # strict generalization of the baseline


def test_widest_basin_prefers_higher_survival_among_neighbors() -> None:
    # two demos at the SAME descriptor; widest_basin must pick the higher-survival theta.
    X = np.array([[0.0, 0.0], [0.0, 0.0]])
    T = np.array([_MID, _MID.copy()])
    T[1, 0] = _MID[0] + 0.01                                        # distinguish the two thetas
    surv = np.array([0.2, 0.9])
    ret = RetrievalDeliveryPolicy.fit(X, T, surv, RetrievalConfig(k=2, select=SelectRule.WIDEST_BASIN))
    assert np.allclose(ret.predict(np.zeros(2)), T[1])             # the wider basin (surv 0.9)


def test_dist_weighted_blends_neighbors() -> None:
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    T = np.array([_MID, _MID + 0.02])
    ret = RetrievalDeliveryPolicy.fit(X, T, np.array([0.5, 0.5]),
                                      RetrievalConfig(standardize=False, k=2, select=SelectRule.DIST_WEIGHTED))
    out = ret.predict(np.array([2.0, 0.0]))                         # between the demos, closer to demo 0
    assert np.all(out >= np.minimum(T[0], T[1])) and np.all(out <= np.maximum(T[0], T[1]))
    assert not np.allclose(out, T[0]) and np.linalg.norm(out - T[0]) < np.linalg.norm(out - T[1])   # nearer demo weighs more


def test_exclude_idx_leave_one_out() -> None:
    X, T, surv = _data()
    ret = RetrievalDeliveryPolicy.fit(X, T, surv, RetrievalConfig(k=1, select=SelectRule.NEAREST))
    # querying with a train point normally returns itself; LOO must return a DIFFERENT demo's theta.
    self_pred = ret.predict(X[2])
    loo_pred = ret.predict(X[2], exclude_idx=2)
    assert np.allclose(self_pred, T[2]) and not np.allclose(loo_pred, T[2])


def test_predict_always_in_box() -> None:
    X, _T, surv = _data()
    T = np.tile(THETA_HI + 5.0, (X.shape[0], 1))                   # out-of-box thetas must be clipped
    ret = RetrievalDeliveryPolicy.fit(X, T, surv, RetrievalConfig())
    out = ret.predict(X[0])
    assert np.all(out <= THETA_HI + 1e-9) and np.all(out >= THETA_LO - 1e-9)


def test_certificate_deployable() -> None:
    c = RetrievalDeploymentCertificate(True, True, True, 1, "nearest", True, 0.61, 0.33, 0.20)
    assert c.is_deployable()
    assert not RetrievalDeploymentCertificate(True, False, True, 1, "nearest", True, 0.6, 0.3, 0.2).is_deployable()
