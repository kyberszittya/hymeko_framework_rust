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

    def to_json(self) -> "dict[str, object]":
        return {"standardize": self.standardize, "k": self.k, "select": self.select.value}

    @staticmethod
    def from_json(d: "dict[str, object]") -> "RetrievalConfig":
        return RetrievalConfig(bool(d["standardize"]), int(d["k"]), SelectRule(str(d["select"])))


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
        return self.predict_with_source(x, exclude_idx)[0]

    def predict_with_source(self, x: np.ndarray, exclude_idx: "int | None" = None) -> "tuple[np.ndarray, int]":
        """Like :meth:`predict`, but also return the NEAREST table-row index — the primary retrieved demonstration —
        so callers can label/audit *which* demo was retrieved without reimplementing the nearest lookup.

        # Postconditions: the theta equals :meth:`predict`; the index is ``argmin`` of the query distance (respecting
          ``exclude_idx``). For ``SelectRule.NEAREST`` the theta IS ``theta[index]``; for k>1 blends the index is the
          nearest (primary) source of the blend."""
        d = self._distances(x, exclude_idx)
        k = min(self._cfg.k, int(np.isfinite(d).sum()))
        idx = np.argsort(d)[:k]
        return clip_theta(self._select(idx, d)), int(idx[0])

    def support_distance(self, x: np.ndarray) -> float:
        """Distance from a query to its NEAREST table row (in the policy's metric) — the retrieval-support signal. A
        large value means the query is outside the demonstrated support (candidate ``RETRIEVAL_OUT_OF_SUPPORT``)."""
        return float(self._distances(x, None).min())

    def table_coverage_radius(self, percentile: float = 95.0) -> float:
        """The table's own coverage radius: the ``percentile`` of each row's distance to its nearest OTHER row. A query
        beyond this is extrapolating past where the demonstrations sit densely. Pure function of the frozen table."""
        n = self._table.shape[0]
        if n < 2:
            return float("inf")
        nn = []
        for i in range(n):
            d = np.linalg.norm(self._table - self._table[i], axis=1)
            d[i] = np.inf
            nn.append(float(d.min()))
        return float(np.percentile(nn, percentile))

    def _select(self, idx: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self._cfg.select is SelectRule.NEAREST:
            return self._theta[int(idx[0])]
        if self._cfg.select is SelectRule.WIDEST_BASIN:
            return self._theta[int(idx[int(np.argmax(self._surv[idx]))])]
        w = 1.0 / (d[idx] + 1e-9)                                     # DIST_WEIGHTED
        w = w / w.sum()
        return (w[:, None] * self._theta[idx]).sum(0)


def freeze_table(scenario_ids: "list[str]", X: np.ndarray, Theta: np.ndarray, survival: np.ndarray,
                 config: RetrievalConfig) -> "dict[str, object]":
    """Serialize a SELF-CONTAINED frozen deployment table (raw descriptors + robust theta + survival + config). The
    table is the whole policy — no dependency on re-deriving it from the bank at load time."""
    return {"scenario_ids": list(scenario_ids),
            "X": np.asarray(X, np.float64).tolist(),
            "theta": np.asarray(Theta, np.float64).tolist(),
            "survival": np.asarray(survival, np.float64).tolist(),
            "config": config.to_json()}


def load_frozen(spec: "dict[str, object]", config: "RetrievalConfig | None" = None) -> "RetrievalDeliveryPolicy":
    """Reconstruct a policy from a frozen table (deterministic re-fit). ``config`` overrides the stored one (e.g. to run
    the in-distribution control on the same table); default uses the frozen config."""
    cfg = config if config is not None else RetrievalConfig.from_json(spec["config"])       # type: ignore[arg-type]
    return RetrievalDeliveryPolicy.fit(np.asarray(spec["X"], np.float64), np.asarray(spec["theta"], np.float64),
                                       np.asarray(spec["survival"], np.float64), cfg)
