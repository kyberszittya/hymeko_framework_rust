"""Tests for R11.4B policies + feature/theta scaling (pure numpy/torch — no MuJoCo)."""
import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_bc.models import (
    THETA_HI,
    THETA_LO,
    MeanThetaPolicy,
    MlpBcPolicy,
    NearestSchedulePolicy,
    RidgePolicy,
    Standardizer,
    _from01,
    _to01,
    clip_theta,
)


def _toy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 30))
    Theta = THETA_LO + rng.random((12, 6)) * (THETA_HI - THETA_LO)
    return X, Theta


def test_clip_theta_projects_into_box() -> None:
    out = clip_theta(np.array([1.0, -1.0, 5.0, 999.0, -5.0, 9.0]))
    assert np.all(out >= THETA_LO) and np.all(out <= THETA_HI)


def test_to01_from01_roundtrip() -> None:
    theta = THETA_LO + 0.37 * (THETA_HI - THETA_LO)
    assert np.allclose(_from01(_to01(theta)), theta, atol=1e-9)


def test_standardizer_constant_feature_gets_unit_std() -> None:
    X = np.hstack([np.random.default_rng(1).standard_normal((8, 2)), np.full((8, 1), 3.0)])
    s = Standardizer.fit(X)
    assert s.std[2] == 1.0                                    # constant column -> std forced to 1, no divide-by-zero
    assert np.allclose(s.transform(X)[:, 2], 0.0)


def test_mean_policy_predicts_train_mean() -> None:
    X, T = _toy()
    p = MeanThetaPolicy.fit(X, T)
    assert np.allclose(p.predict(X[0]), clip_theta(T.mean(0)))
    assert np.allclose(p.predict(X[3]), p.predict(X[0]))       # ignores the descriptor


def test_nearest_schedule_returns_own_theta_at_zero_distance() -> None:
    X, T = _toy()
    p = NearestSchedulePolicy.fit(X, T)
    assert np.allclose(p.predict(X[5]), clip_theta(T[5]))
    assert p.nn_distance(X[5]) == pytest.approx(0.0, abs=1e-9)


def test_ridge_predicts_in_box_and_high_lambda_shrinks_to_mean() -> None:
    X, T = _toy()
    pred = RidgePolicy.fit(X, T, lam=1.0).predict(X[2])
    assert np.all(pred >= THETA_LO) and np.all(pred <= THETA_HI)
    big = RidgePolicy.fit(X, T, lam=1e9)
    assert np.allclose(big.predict(X[0]), big.predict(X[7]), atol=1e-3)   # lambda->inf => bias only => constant


def test_mlp_fit_predict_shape_and_box() -> None:
    X, T = _toy()
    p = MlpBcPolicy.fit(X, T, epochs=40, seed=0)
    out = p.predict(X[1])
    assert out.shape == (6,) and np.all(out >= THETA_LO) and np.all(out <= THETA_HI)
    assert np.allclose(p.predict(X[1]), MlpBcPolicy.fit(X, T, epochs=40, seed=0).predict(X[1]))  # deterministic
