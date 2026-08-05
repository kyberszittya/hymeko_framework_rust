"""Nearest-robust-basin RETRIEVAL delivery policy — the teacher-free deployment form indicated by the R11.5R density
curve (retrieval is density-responsive where the smooth ridge/mlp regressors are descriptor-limited).

At run time the policy uses ONLY a stored table of (descriptor, robust theta, survival) and a nearest lookup: no CEM, no
oracle, no teacher. It is a strict generalization of ``NearestSchedulePolicy`` — ``RetrievalConfig(standardize=True,
k=1, select=NEAREST)`` reproduces it exactly (pinned by a parity test). The two design axes (the descriptor metric and
the neighborhood/tie-break rule) are a config, not a Cartesian product of functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer, clip_theta


class SelectRule(Enum):
    """How to turn the k nearest demonstrations into one theta."""

    NEAREST = "nearest"              # the single closest demo (k-independent; == NearestSchedulePolicy)
    WIDEST_BASIN = "widest_basin"    # among the k nearest, the highest-survival (widest-basin) theta
    DIST_WEIGHTED = "dist_weighted"  # inverse-distance-weighted mean theta over the k nearest


@dataclass(frozen=True)
class RetrievalConfig:
    standardize: bool = True
    k: int = 1
    select: SelectRule = SelectRule.NEAREST


@dataclass(frozen=True)
class RetrievalDeploymentCertificate:
    """A retrieval policy is a TEACHER-FREE deployment: no CEM, no oracle, no teacher at run time — only a stored table
    and a nearest lookup. ``coverage_*`` are closed-loop strict-K6 rates per split (train is leave-one-out)."""

    teacher_free: bool
    cem_free: bool
    oracle_free: bool
    k: int
    select: str
    standardized: bool
    coverage_train_loo: float
    coverage_dev: float
    coverage_test: float

    def is_deployable(self) -> bool:
        """A retrieval policy is deployable iff it needs no teacher-time search or oracle."""
        return self.teacher_free and self.cem_free and self.oracle_free


class RetrievalDeliveryPolicy:
    """descriptor -> k nearest robust demos -> one theta by the select rule -> clip to the certified box.

    Preconditions: ``X`` (N, F) descriptors, ``Theta`` (N, 6) certified robust thetas, ``survival`` (N,) local-K6
    survival in [0, 1]; N >= 1. Postconditions: ``predict`` returns a theta inside the certified box.
    """

    name = "retrieval"

    def __init__(self, table: np.ndarray, thetas: np.ndarray, survival: np.ndarray, config: RetrievalConfig,
                 std: "Standardizer | None") -> None:
        self._table = table          # (N, F) descriptors in the query metric (standardized or raw)
        self._theta = thetas         # (N, 6)
        self._surv = survival        # (N,)
        self._cfg = config
        self._std = std

    @staticmethod
    def fit(X: np.ndarray, Theta: np.ndarray, survival: np.ndarray,
            config: RetrievalConfig = RetrievalConfig()) -> "RetrievalDeliveryPolicy":
        X = np.atleast_2d(np.asarray(X, np.float64))
        std = Standardizer.fit(X) if config.standardize else None
        table = std.transform(X) if std is not None else X
        return RetrievalDeliveryPolicy(table, np.asarray(Theta, np.float64), np.asarray(survival, np.float64), config, std)

    def _distances(self, x: np.ndarray, exclude_idx: "int | None") -> np.ndarray:
        q = self._std.transform(x) if self._std is not None else np.atleast_2d(np.asarray(x, np.float64))
        d = np.linalg.norm(self._table - q, axis=1)
        if exclude_idx is not None:
            d = d.copy()
            d[exclude_idx] = np.inf                                   # leave-one-out: never retrieve self
        return d

    def predict(self, x: np.ndarray, exclude_idx: "int | None" = None) -> np.ndarray:
        """Map one descriptor to a clipped theta. ``exclude_idx`` drops that table row (leave-one-out train eval)."""
        d = self._distances(x, exclude_idx)
        k = min(self._cfg.k, int(np.isfinite(d).sum()))
        idx = np.argsort(d)[:k]
        return clip_theta(self._select(idx, d))

    def _select(self, idx: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self._cfg.select is SelectRule.NEAREST:
            return self._theta[int(idx[0])]
        if self._cfg.select is SelectRule.WIDEST_BASIN:
            return self._theta[int(idx[int(np.argmax(self._surv[idx]))])]
        w = 1.0 / (d[idx] + 1e-9)                                     # DIST_WEIGHTED
        w = w / w.sum()
        return (w[:, None] * self._theta[idx]).sum(0)
