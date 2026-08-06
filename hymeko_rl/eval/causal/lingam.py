"""DirectLiNGAM over continuous rollout variables — numpy/scipy only, no external ``lingam`` dependency.

DirectLiNGAM (Shimizu et al. 2011, *A direct method for learning a linear non-Gaussian structural equation
model*) recovers a causal ordering of variables under the LiNGAM assumptions — linear structural equations,
non-Gaussian independent disturbances, an acyclic structure, and causal sufficiency — **without** iterative ICA.
It is the exploratory causal-discovery step of the CIP diagnostic layer.

**This module PROPOSES candidate structure; it never proves causality.** Controlled ablations decide (framework
doctrine — the causal-discovery instance of the discriminating-test rule). The LiNGAM assumptions are routinely
violated by contact dynamics (non-linear) and control loops (cyclic); a recovered edge is a hypothesis to test,
not a verdict. Callers must treat :class:`LingamResult` as a ranked set of interventions to try, not a truth.

Design (paradigm hierarchy: Strategy trait + pure numerical core):

* :class:`IndependenceMeasure` — the swappable pairwise independence statistic (Strategy). The default
  :class:`PairwiseEntropyMeasure` is the Hyvärinen maximum-entropy approximation used by the reference
  DirectLiNGAM; a different measure (e.g. an HSIC/kernel measure) is a drop-in without touching the search.
* :class:`DirectLiNGAM` — the ordering search + adjacency estimation; holds no global state, threads config
  explicitly (§6.5 #11 no globals).

Only continuous, roughly-standardisable columns belong here. Categorical strata (method / architecture / seed /
phase) must be stratified upstream, never fed into the linear model — the caller (:mod:`.diagnose`) enforces that;
this module assumes its input is already continuous-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy import stats  # type: ignore[import-untyped]  # scipy ships no stubs; it is a pinned repo dependency

# Hyvärinen (1998) maximum-entropy approximation constants (differential entropy of a unit-variance signal).
_K1 = 79.047
_K2 = 7.4129
_GAMMA = 0.37457
_HALF_LOG_2PI_E = (1.0 + float(np.log(2.0 * np.pi))) / 2.0
_EPS = 1e-12


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance copy of a 1-D signal (std floored by ``_EPS`` to avoid divide-by-zero)."""
    x = np.asarray(x, dtype=np.float64)
    result: np.ndarray = (x - x.mean()) / (x.std() + _EPS)
    return result


def _entropy(u: np.ndarray) -> float:
    """Differential-entropy estimate of a (near) unit-variance signal via the Hyvärinen max-entropy approximation.

    # Preconditions
    * ``u`` is standardised (zero mean, unit variance) for the approximation to be calibrated.
    """
    log_cosh = float(np.mean(np.log(np.cosh(u))))
    gauss = float(np.mean(u * np.exp(-(u ** 2) / 2.0)))
    return _HALF_LOG_2PI_E - _K1 * (log_cosh - _GAMMA) ** 2 - _K2 * gauss ** 2


def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """Linear residual of ``xi`` after regressing out ``xj`` (both standardised): ``xi - corr(xi,xj) * xj``."""
    cov = float(np.mean(xi * xj))          # == correlation for unit-variance standardised inputs
    return xi - cov * xj


def _ols_with_pvalues(predictors: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and two-sided t-test p-values for each predictor (``predictors`` shape ``(n, k)``).

    # Postconditions
    * Returns ``(coef[k], pvalues[k])``; a predictor with insufficient residual degrees of freedom is reported
      as significant (``p = 0``) so it is not spuriously dropped.
    """
    n, k = predictors.shape
    coef, *_ = np.linalg.lstsq(predictors, y, rcond=None)
    resid = y - predictors @ coef
    dof = n - k
    if dof <= 0:
        return coef, np.zeros(k)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(predictors.T @ predictors)
    var = np.maximum(np.diag(xtx_inv) * sigma2, 0.0)
    se = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0.0, coef / se, np.inf)
    pvals: np.ndarray = 2.0 * stats.t.sf(np.abs(tvals), dof)
    return coef, pvals


def sample_linear_sem(
    edges: Sequence[tuple[int, int, float]],
    n_vars: int,
    n_samples: int,
    seed: int,
    noise: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_samples`` rows from the linear non-Gaussian SEM ``x = Bx + e`` with a known DAG.

    ``edges`` are ``(cause, effect, weight)`` triples defining ``B[effect, cause] = weight``. Disturbances are
    unit-variance uniform or Laplace (both non-Gaussian, satisfying the LiNGAM identifiability condition).
    Returns ``(X[n_samples, n_vars], B[n_vars, n_vars])``. Used by the discovery tests and the demo to provide
    ground-truth causal structure.

    # Preconditions
    * ``edges`` form a DAG (else ``I - B`` may be singular); indices are in ``range(n_vars)``.
    """
    rng = np.random.default_rng(seed)
    if noise == "uniform":
        e = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n_samples, n_vars))
    elif noise == "laplace":
        e = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(n_samples, n_vars))
    else:
        raise ValueError(f"sample_linear_sem: noise must be 'uniform' or 'laplace', got {noise!r}")
    b = np.zeros((n_vars, n_vars), dtype=np.float64)
    for cause, effect, weight in edges:
        b[effect, cause] = weight
    x: np.ndarray = e @ np.linalg.inv(np.eye(n_vars) - b).T
    return x, b


@runtime_checkable
class IndependenceMeasure(Protocol):
    """Strategy: a signed pairwise statistic that is ``>= 0`` when ``xj`` is the more exogenous of the pair.

    ``diff(xi, xj)`` estimates the difference in mutual information between the two candidate directions
    ``xj -> xi`` and ``xi -> xj``. It is ``>= 0`` when ``xj`` can be a cause of ``xi``; the search sums
    ``min(0, diff)^2`` over partners, so a true root accumulates zero penalty.
    """

    def diff(self, xi: np.ndarray, xj: np.ndarray) -> float:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class PairwiseEntropyMeasure:
    """Default measure: entropy-based difference of mutual information (the reference DirectLiNGAM statistic)."""

    def diff(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """``(H(xj) + H(res(xi|xj))) - (H(xi) + H(res(xj|xi)))`` on standardised residuals; ``>=0`` if xj exogenous.

        # Preconditions
        * ``xi``, ``xj`` are standardised 1-D arrays of equal length.
        # Postconditions
        * Returns a finite float; sign encodes the more-exogenous direction (``>= 0`` ⇒ ``xj`` may cause ``xi``).
        """
        ri_j = _standardize(_residual(xi, xj))
        rj_i = _standardize(_residual(xj, xi))
        return (_entropy(xj) + _entropy(ri_j)) - (_entropy(xi) + _entropy(rj_i))


@dataclass(frozen=True)
class LingamConfig:
    """Configuration for :class:`DirectLiNGAM`.

    ``measure`` — the pairwise independence Strategy. ``alpha`` — significance level for backward elimination of
    adjacency edges: a predecessor whose partial-correlation p-value exceeds ``alpha`` is dropped (this removes
    transitive/indirect leakage that a raw magnitude threshold cannot, since an ancestor is insignificant once
    the true direct parent is conditioned on). ``prune_threshold`` — a magnitude floor applied after significance
    pruning (tiny surviving coefficients are zeroed). ``standardize_adjacency`` — estimate on standardised columns
    (coefficients comparable across differently-scaled variables).
    """

    measure: IndependenceMeasure = field(default_factory=PairwiseEntropyMeasure)
    alpha: float = 0.01
    prune_threshold: float = 0.01
    standardize_adjacency: bool = True


@dataclass(frozen=True)
class LingamResult:
    """Recovered causal ordering and adjacency. **Proposed structure, not proof.**

    ``order`` — variable indices from most exogenous to most endogenous. ``adjacency`` — ``B`` with
    ``x_k = sum_j B[k, j] x_j + e_k`` (strictly triangular under ``order``). ``names`` — column names.
    """

    order: list[int]
    adjacency: np.ndarray
    names: list[str]

    def ordered_names(self) -> list[str]:
        """Variable names in discovered causal order (cause → effect)."""
        return [self.names[i] for i in self.order]

    def strongest_edges(self, k: int = 5) -> list[tuple[str, str, float]]:
        """Top-``k`` ``(cause, effect, weight)`` by ``abs`` coefficient. Proposed edges, ranked for ablation."""
        edges = [(self.names[j], self.names[i], float(self.adjacency[i, j]))
                 for i in range(len(self.names)) for j in range(len(self.names))
                 if self.adjacency[i, j] != 0.0]
        edges.sort(key=lambda e: -abs(e[2]))
        return edges[:k]


class DirectLiNGAM:
    """DirectLiNGAM ordering search + least-squares adjacency estimation (numpy/scipy only).

    # Preconditions
    * ``fit`` receives ``X`` of shape ``(n_samples, n_vars)`` with ``n_samples > n_vars``, finite, and at least
      two non-constant columns.
    # Postconditions
    * Returns a :class:`LingamResult` whose ``order`` is a permutation of ``range(n_vars)`` and whose
      ``adjacency`` is strictly triangular in that order (no self-loops, acyclic by construction).
    # Invariants
    * Holds no mutable state across ``fit`` calls; the config is immutable.
    """

    def __init__(self, config: LingamConfig | None = None) -> None:
        self.config = config or LingamConfig()

    def fit(self, x: np.ndarray, names: list[str] | None = None) -> LingamResult:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"DirectLiNGAM.fit: X must be 2-D (n_samples, n_vars), got shape {x.shape}")
        n, d = x.shape
        if d < 2:
            raise ValueError(f"DirectLiNGAM.fit: need >= 2 variables, got {d}")
        if n <= d:
            raise ValueError(f"DirectLiNGAM.fit: need n_samples > n_vars, got n={n}, d={d}")
        if not np.all(np.isfinite(x)):
            raise ValueError("DirectLiNGAM.fit: X contains non-finite values (clean/drop before fitting)")
        names = list(names) if names is not None else [f"x{i}" for i in range(d)]
        if len(names) != d:
            raise ValueError(f"DirectLiNGAM.fit: names has {len(names)} entries, expected {d}")

        order = self._search_causal_order(x)
        adjacency = self._estimate_adjacency(x, order)
        return LingamResult(order=order, adjacency=adjacency, names=names)

    # -- ordering search --------------------------------------------------------------------------------------
    def _search_causal_order(self, x: np.ndarray) -> list[int]:
        """Greedy most-exogenous-first search. Returns the full causal order (root → leaf)."""
        d = x.shape[1]
        remaining = list(range(d))
        residual_cols = {i: _standardize(x[:, i]) for i in remaining}
        order: list[int] = []
        while len(remaining) > 1:
            root = self._most_exogenous(remaining, residual_cols)
            order.append(root)
            remaining.remove(root)
            # Regress the chosen root out of every remaining column, then re-standardise for the next round.
            residual_cols = {i: _standardize(_residual(residual_cols[i], residual_cols[root]))
                             for i in remaining}
        order.append(remaining[0])
        return order

    def _most_exogenous(self, remaining: list[int], cols: dict[int, np.ndarray]) -> int:
        """Index in ``remaining`` minimising the accumulated non-exogeneity penalty ``sum min(0, diff)^2``."""
        measure = self.config.measure
        best_idx = remaining[0]
        best_penalty = np.inf
        for cand in remaining:
            penalty = 0.0
            for other in remaining:
                if other == cand:
                    continue
                diff = measure.diff(cols[cand], cols[other])
                neg = min(0.0, diff)
                penalty += neg * neg
            if penalty < best_penalty:
                best_penalty = penalty
                best_idx = cand
        return best_idx

    # -- adjacency estimation ---------------------------------------------------------------------------------
    def _estimate_adjacency(self, x: np.ndarray, order: list[int]) -> np.ndarray:
        """Regress each variable on its predecessors with backward elimination by significance, then a floor.

        For each variable, start from all predecessors in ``order`` and repeatedly drop the least-significant one
        (largest partial-correlation p-value) while it exceeds ``alpha``; then zero any surviving coefficient below
        ``prune_threshold``. This recovers *direct* parents only — transitive ancestors become insignificant once
        the direct parent is conditioned on.
        """
        d = x.shape[1]
        data = x.copy()
        if self.config.standardize_adjacency:
            data = np.column_stack([_standardize(data[:, i]) for i in range(d)])
        b = np.zeros((d, d), dtype=np.float64)
        for pos, target in enumerate(order):
            active = list(order[:pos])
            coef = self._backward_eliminate(data, target, active)
            for p_idx, coefficient in coef.items():
                b[target, p_idx] = coefficient
        return b

    def _backward_eliminate(self, data: np.ndarray, target: int, active: list[int]) -> dict[int, float]:
        """Drop insignificant predecessors one at a time; return the surviving ``{predecessor: coefficient}``."""
        y = data[:, target]
        while active:
            coef, pvals = _ols_with_pvalues(data[:, active], y)
            worst = int(np.argmax(pvals))
            if pvals[worst] > self.config.alpha:
                active.pop(worst)
                continue
            keep = {p: float(c) for p, c in zip(active, coef) if abs(c) >= self.config.prune_threshold}
            if len(keep) == len(active):
                return keep
            active = list(keep)          # a coefficient fell below the magnitude floor → refit without it
        return {}
