"""R11.4B policies — the ``ThetaPolicy`` Protocol and its implementations, plus the mandatory simple controls.

Every policy maps a conditioning descriptor to a theta in R^6, clipped to the certified transport box. The small MLP runs
ALONGSIDE the controls (global mean-theta, nearest-schedule, ridge); if a control matches it, the report says so plainly.
Ridge and the MLP regress a box-normalized theta (theta dims span two orders of magnitude — normalization is what makes
their squared-error objective sane); mean and nearest-schedule copy raw theta and need no normalization.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec

_SPEC = full_transport_spec()
THETA_LO = np.asarray(_SPEC.lo, np.float64)     # (6,) certified transport box, single source of truth
THETA_HI = np.asarray(_SPEC.hi, np.float64)


def clip_theta(theta: np.ndarray) -> np.ndarray:
    """Project a predicted theta into the certified box so no out-of-contract actuation ever reaches the primitive."""
    return np.clip(np.asarray(theta, np.float64), THETA_LO, THETA_HI)


def _to01(theta: np.ndarray) -> np.ndarray:
    return (np.asarray(theta, np.float64) - THETA_LO) / (THETA_HI - THETA_LO)


def _from01(z: np.ndarray) -> np.ndarray:
    return clip_theta(THETA_LO + np.asarray(z, np.float64) * (THETA_HI - THETA_LO))


class Standardizer:
    """Zero-mean/unit-std feature scaling fit on the TRAIN split only (std=1 where a feature is constant)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        X = np.atleast_2d(np.asarray(X, np.float64))
        std = X.std(0)
        std[std < 1e-9] = 1.0
        return Standardizer(X.mean(0), std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, np.float64) - self.mean) / self.std


@runtime_checkable
class ThetaPolicy(Protocol):
    name: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Map one descriptor (shape ``(N_FEATURES,)``) to a clipped theta (shape ``(6,)``)."""
        ...


class MeanThetaPolicy:
    """Control: predict the train-set mean theta for every scenario (ignores the descriptor)."""

    def __init__(self, theta_mean: np.ndarray) -> None:
        self.name = "mean_theta"
        self._theta = clip_theta(theta_mean)

    @staticmethod
    def fit(X: np.ndarray, Theta: np.ndarray) -> "MeanThetaPolicy":
        return MeanThetaPolicy(np.asarray(Theta, np.float64).mean(0))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._theta.copy()


class NearestSchedulePolicy:
    """Control: 1-NN in standardized descriptor space — copy the nearest train scenario's certified theta."""

    def __init__(self, std: Standardizer, Xs: np.ndarray, Theta: np.ndarray) -> None:
        self.name = "nearest_schedule"
        self._std = std
        self._Xs = Xs
        self._Theta = Theta

    @staticmethod
    def fit(X: np.ndarray, Theta: np.ndarray) -> "NearestSchedulePolicy":
        std = Standardizer.fit(X)
        return NearestSchedulePolicy(std, std.transform(X), np.asarray(Theta, np.float64))

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(self._Xs - self._std.transform(x), axis=1)
        return clip_theta(self._Theta[int(np.argmin(d))])

    def nn_distance(self, x: np.ndarray) -> float:
        """Standardized distance to the nearest train neighbour — the BC_DATA_COVERAGE diagnostic."""
        return float(np.linalg.norm(self._Xs - self._std.transform(x), axis=1).min())


class RidgePolicy:
    """Control: L2-regularized linear map (standardized descriptor -> box-normalized theta), closed-form."""

    def __init__(self, name: str, std: Standardizer, W: np.ndarray, b: np.ndarray) -> None:
        self.name = name
        self._std = std
        self._W = W
        self._b = b

    @staticmethod
    def fit(X: np.ndarray, Theta: np.ndarray, lam: float = 1.0) -> "RidgePolicy":
        std = Standardizer.fit(X)
        Xs = std.transform(X)
        A = np.hstack([Xs, np.ones((Xs.shape[0], 1))])            # bias column
        Z = _to01(Theta)
        reg = lam * np.eye(A.shape[1])
        reg[-1, -1] = 0.0                                         # do not regularize the bias
        Wf = np.linalg.solve(A.T @ A + reg, A.T @ Z)
        return RidgePolicy("ridge", std, Wf[:-1], Wf[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _from01(self._std.transform(x) @ self._W + self._b)


class MlpBcPolicy:
    """The neural BC: a small MLP (2 hidden layers) regressing box-normalized theta. Small on purpose (~50 demos)."""

    def __init__(self, std: Standardizer, net: Any) -> None:
        self.name = "mlp_bc"
        self._std = std
        self._net = net

    @staticmethod
    def fit(X: np.ndarray, Theta: np.ndarray, hidden: int = 64, epochs: int = 600, lr: float = 3e-3,
            weight_decay: float = 1e-3, seed: int = 0) -> "MlpBcPolicy":
        import torch
        from torch import nn

        torch.manual_seed(seed)
        std = Standardizer.fit(X)
        xt = torch.tensor(std.transform(X), dtype=torch.float32)
        zt = torch.tensor(_to01(Theta), dtype=torch.float32)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.Tanh(),
                            nn.Linear(hidden, hidden), nn.Tanh(),
                            nn.Linear(hidden, zt.shape[1]), nn.Sigmoid())
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(net(xt), zt).backward()
            opt.step()
        net.eval()
        return MlpBcPolicy(std, net)

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            xs = torch.tensor(self._std.transform(np.atleast_2d(x)), dtype=torch.float32)
            z = self._net(xs).numpy()[0]
        return _from01(z)
