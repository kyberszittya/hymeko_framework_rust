"""Tests for the R11.4B basin-robustness diagnostic (rollout monkeypatched — no MuJoCo)."""
import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_bc import evaluate as E
from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO


def test_theta_basin_survival_monotone_in_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    theta0 = np.array([0.1, 0.2, 0.0, 15.0, 30.0, 1.5])
    span = THETA_HI - THETA_LO

    def fake_rollout(snap: object, theta: np.ndarray, cfg: object = None) -> dict:
        z = np.linalg.norm((np.asarray(theta) - theta0) / span)          # box-normalized distance from the certified theta
        return {"k6": bool(z < 0.02), "safe": True, "dtz_mm": 5.0}

    monkeypatch.setattr(E, "rollout_theta", fake_rollout)
    surv = E.theta_basin_survival(None, theta0, (0.001, 0.05), k=8, seed=0)
    assert surv[0.001] >= surv[0.05]                                     # tighter perturbation survives at least as often
    assert surv[0.001] == 1.0 and surv[0.05] <= 0.5                      # narrow (radius-0.02) basin
    assert E.theta_basin_survival(None, theta0, (0.001,), k=8, seed=0) == \
        E.theta_basin_survival(None, theta0, (0.001,), k=8, seed=0)      # deterministic given seed
