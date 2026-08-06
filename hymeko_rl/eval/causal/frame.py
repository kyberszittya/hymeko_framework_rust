"""RolloutFrame — data-oriented (struct-of-arrays) aggregation of per-rollout CIP + continuous variables.

One row per rollout; columns are split by kind so the continuous-only contract of DirectLiNGAM is structural,
not a runtime hope:

* **continuous** columns (float arrays) — the only columns that may enter the linear non-Gaussian model.
* **categorical** columns (object lists) — binary monitor flags and true categoricals (method / architecture /
  seed / phase); these *stratify*, they never mix into the linear model.
* **missing** masks — per CIP variable, ``True`` where the value was defaulted by the export bridge rather than
  measured (a consumer drops all-missing columns before fitting; a default is never presented as a measurement).

The CIP variables are sourced through :func:`hymeko_rl.eval.task_monitor.cip_export.export_cip_variables`
(§6.1 — the disagreement/monitor values are read, never re-derived here). Extra continuous rollout variables
(reward, distances, sub-scores) and categorical strata are supplied column-wise by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, Sized, TypeVar

import numpy as np

from hymeko_rl.eval.task_monitor.cip_export import CIP_VARIABLES, CipExport, export_cip_variables

_Col = TypeVar("_Col", bound=Sized)


class VarKind(Enum):
    """How a variable may be used. Only :attr:`CONTINUOUS` may enter the linear causal model."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"     # binary monitor flags + true categoricals → stratify, never mix in


# Kind of each exported CIP variable. Booleans are categorical (degenerate for a linear model); counts and
# scores are continuous (treated as continuous iff they show variance — see :meth:`RolloutFrame.continuous_matrix`).
CIP_VAR_KINDS: dict[str, VarKind] = {
    "success_monitor_pass": VarKind.CATEGORICAL,
    "progress_score": VarKind.CONTINUOUS,
    "stagnation_duration": VarKind.CONTINUOUS,
    "stagnated": VarKind.CATEGORICAL,
    "forbidden_contact_count": VarKind.CONTINUOUS,
    "clearance_min": VarKind.CONTINUOUS,
    "phase_transition_failure": VarKind.CATEGORICAL,
    "reward_progress_disagreement": VarKind.CONTINUOUS,
}
assert set(CIP_VAR_KINDS) == set(CIP_VARIABLES), "CIP_VAR_KINDS must classify every CIP variable"


def _checked_column(col: _Col, name: str, n: int, source: str) -> _Col:
    """Return ``col`` unchanged after asserting its length is ``n`` (raise with the offending source otherwise)."""
    if len(col) != n:
        raise ValueError(f"{source}[{name!r}] length {len(col)} != n={n}")
    return col


def _split_cip_variables(
    exports: Sequence[CipExport],
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]], dict[str, np.ndarray]]:
    """Split exported CIP variables into continuous / categorical columns + a missing mask per continuous var."""
    continuous: dict[str, np.ndarray] = {}
    categorical: dict[str, list[Any]] = {}
    missing: dict[str, np.ndarray] = {}
    for name in CIP_VARIABLES:
        values = [e.variables[name] for e in exports]
        if CIP_VAR_KINDS[name] is VarKind.CONTINUOUS:
            continuous[name] = np.asarray(values, dtype=np.float64)
            missing[name] = np.array([name in e.missing for e in exports], dtype=bool)
        else:
            categorical[name] = [bool(x) for x in values]
    return continuous, categorical, missing


@dataclass
class RolloutFrame:
    """Column-oriented per-rollout variables. See module docstring for the continuous/categorical/missing split."""

    continuous: dict[str, np.ndarray]
    categorical: dict[str, list[Any]]
    missing: dict[str, np.ndarray] = field(default_factory=dict)
    n: int = 0

    def __post_init__(self) -> None:
        for name, col in self.continuous.items():
            if len(col) != self.n:
                raise ValueError(f"continuous column {name!r} has length {len(col)}, expected n={self.n}")
        for name, cat_col in self.categorical.items():
            if len(cat_col) != self.n:
                raise ValueError(f"categorical column {name!r} has length {len(cat_col)}, expected n={self.n}")

    # -- construction -----------------------------------------------------------------------------------------
    @classmethod
    def from_verdicts(
        cls,
        verdicts: Sequence[Any],
        extra_continuous: Mapping[str, Sequence[float]] | None = None,
        categoricals: Mapping[str, Sequence[Any]] | None = None,
    ) -> "RolloutFrame":
        """Build a frame from per-rollout monitor verdicts plus caller-supplied continuous/categorical columns.

        # Preconditions
        * ``verdicts`` is a non-empty sequence of ``TaskVerdict`` objects or their ``as_dict()`` mappings.
        * every column in ``extra_continuous`` / ``categoricals`` has length ``len(verdicts)``.
        # Postconditions
        * CIP variables are split into continuous/categorical per :data:`CIP_VAR_KINDS`; a defaulted CIP value is
          flagged in :attr:`missing` and never silently treated as a measurement.
        """
        n = len(verdicts)
        if n == 0:
            raise ValueError("RolloutFrame.from_verdicts: need at least one verdict")

        continuous, categorical, missing = _split_cip_variables([export_cip_variables(v) for v in verdicts])
        for name, col in (extra_continuous or {}).items():
            continuous[name] = _checked_column(np.asarray(list(col), dtype=np.float64), name, n, "extra_continuous")
        for name, col in (categoricals or {}).items():
            categorical[name] = list(_checked_column(list(col), name, n, "categoricals"))

        return cls(continuous=continuous, categorical=categorical, missing=missing, n=n)

    # -- selection --------------------------------------------------------------------------------------------
    def continuous_matrix(
        self,
        names: Sequence[str] | None = None,
        drop_constant: bool = True,
        min_std: float = 1e-9,
    ) -> tuple[np.ndarray, list[str], dict[str, str]]:
        """Stack usable continuous columns into a matrix, dropping all-missing/constant ones (with a reason).

        Returns ``(matrix[n, k], kept_names, dropped)`` where ``dropped`` maps a dropped name to its reason
        (``"all_missing"`` / ``"constant"`` / ``"unknown"``). Requested ``names`` that are categorical raise —
        a categorical must never reach the linear model.
        """
        requested = list(names) if names is not None else list(self.continuous)
        kept: list[str] = []
        cols: list[np.ndarray] = []
        dropped: dict[str, str] = {}
        for name in requested:
            if name in self.categorical:
                raise ValueError(
                    f"continuous_matrix: {name!r} is categorical; stratify on it, do not feed it to the model")
            if name not in self.continuous:
                dropped[name] = "unknown"
                continue
            col = self.continuous[name]
            miss = self.missing.get(name)
            if miss is not None and bool(miss.all()):
                dropped[name] = "all_missing"
                continue
            if drop_constant and float(np.std(col)) < min_std:
                dropped[name] = "constant"
                continue
            kept.append(name)
            cols.append(col)
        matrix = np.column_stack(cols) if cols else np.empty((self.n, 0))
        return matrix, kept, dropped

    # -- stratification ---------------------------------------------------------------------------------------
    def group_by(self, keys: Sequence[str]) -> dict[tuple[Any, ...], np.ndarray]:
        """Group row indices by the tuple of categorical ``keys`` (the stratification the LiNGAM step honours).

        # Preconditions
        * every key is a categorical column present in the frame.
        """
        for k in keys:
            if k not in self.categorical:
                raise ValueError(f"group_by: {k!r} is not a categorical column ({list(self.categorical)})")
        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(self.n):
            label = tuple(self.categorical[k][i] for k in keys)
            groups.setdefault(label, []).append(i)
        return {label: np.asarray(idx, dtype=int) for label, idx in groups.items()}

    def subset(self, indices: Iterable[int]) -> "RolloutFrame":
        """A new frame over ``indices`` (used to run the model within a stratum)."""
        idx = np.asarray(list(indices), dtype=int)
        return RolloutFrame(
            continuous={k: v[idx] for k, v in self.continuous.items()},
            categorical={k: [v[i] for i in idx] for k, v in self.categorical.items()},
            missing={k: v[idx] for k, v in self.missing.items()},
            n=len(idx),
        )
