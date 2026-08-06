"""Correctness tests for the numpy/scipy DirectLiNGAM (the discriminating test for the entropy measure).

A subtly wrong independence statistic fails the ground-truth recovery test below: we synthesise a linear
non-Gaussian SEM with a *known* DAG and assert DirectLiNGAM recovers (i) a valid topological order and
(ii) the adjacency support with correct signs. Plus input-contract guards and determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.causal.lingam import (
    DirectLiNGAM,
    IndependenceMeasure,
    LingamConfig,
    PairwiseEntropyMeasure,
    sample_linear_sem,
)


def _gen_sem(edges, d, n, seed, noise="uniform"):
    """Thin alias over the package's ground-truth SEM sampler (see :func:`sample_linear_sem`)."""
    return sample_linear_sem(edges, d, n, seed, noise)


def _positions(order):
    return {v: i for i, v in enumerate(order)}


def test_recovers_topological_order_and_signs_uniform():
    edges = [(0, 1, 0.8), (0, 2, -0.6), (1, 3, 0.5), (2, 3, 0.7)]
    x, _ = _gen_sem(edges, d=4, n=3000, seed=0, noise="uniform")
    res = DirectLiNGAM().fit(x, ["a", "b", "c", "d"])
    pos = _positions(res.order)
    for cause, effect, w in edges:
        assert pos[cause] < pos[effect]                         # cause precedes effect
        assert res.adjacency[effect, cause] != 0.0             # true edge recovered
        assert np.sign(res.adjacency[effect, cause]) == np.sign(w)   # correct sign
    assert sorted(res.order) == [0, 1, 2, 3]                   # order is a permutation
    # adjacency is strictly triangular under the recovered order (acyclic, no self-loop)
    for i in range(4):
        assert res.adjacency[i, i] == 0.0
        for j in range(4):
            if res.adjacency[i, j] != 0.0:
                assert pos[j] < pos[i]


def test_recovers_under_scrambled_labels_laplace():
    # A fork+collider whose root is NOT column 0, so a trivial "return sorted indices" would fail.
    edges = [(2, 0, 0.9), (2, 1, 0.7), (0, 3, 0.6), (1, 3, -0.5)]
    x, _ = _gen_sem(edges, d=4, n=4000, seed=7, noise="laplace")
    res = DirectLiNGAM().fit(x)
    pos = _positions(res.order)
    assert res.order[0] == 2                                   # variable 2 is the unique root
    for cause, effect, w in edges:
        assert pos[cause] < pos[effect]
        assert np.sign(res.adjacency[effect, cause]) == np.sign(w)


def test_significance_pruning_removes_transitive_edges():
    # Pure chain a->b->c->d: only the three DIRECT edges survive. Naive lstsq-on-all-predecessors would leak
    # transitive edges (a->c, a->d, b->d); backward elimination by significance removes them (regression test).
    edges = [(0, 1, 0.8), (1, 2, 0.8), (2, 3, 0.8)]
    x, _ = _gen_sem(edges, d=4, n=4000, seed=11)
    res = DirectLiNGAM().fit(x, list("abcd"))
    nz = {(res.names[j], res.names[i]) for i in range(4) for j in range(4) if res.adjacency[i, j] != 0.0}
    assert nz == {("a", "b"), ("b", "c"), ("c", "d")}


def test_ols_with_pvalues_guards_zero_dof():
    from hymeko_rl.eval.causal.lingam import _ols_with_pvalues
    # n <= k (no residual degrees of freedom) -> reported significant (p == 0), never spuriously dropped
    coef, pvals = _ols_with_pvalues(np.eye(3), np.array([1.0, 2.0, 3.0]))
    assert pvals.shape == (3,) and np.all(pvals == 0.0)
    # well-posed case: a strong predictor is significant, pure noise is not
    rng = np.random.default_rng(0)
    xp = rng.normal(size=(500, 2))
    y = 1.5 * xp[:, 0] + 0.01 * rng.normal(size=500)
    coef, pvals = _ols_with_pvalues(xp, y)
    assert pvals[0] < 1e-6 and pvals[1] > 0.01


def test_strongest_edges_ranks_by_abs_weight():
    edges = [(0, 1, 0.8), (0, 2, -0.6), (1, 3, 0.5), (2, 3, 0.7)]
    x, _ = _gen_sem(edges, d=4, n=3000, seed=1)
    res = DirectLiNGAM().fit(x, ["a", "b", "c", "d"])
    top = res.strongest_edges(k=2)
    assert len(top) == 2
    assert abs(top[0][2]) >= abs(top[1][2])                    # sorted descending by |weight|
    assert all(isinstance(t[0], str) and isinstance(t[1], str) for t in top)


def test_determinism_same_seed_same_result():
    edges = [(0, 1, 0.8), (1, 2, 0.6)]
    x, _ = _gen_sem(edges, d=3, n=2000, seed=3)
    a = DirectLiNGAM().fit(x)
    b = DirectLiNGAM().fit(x)
    assert a.order == b.order
    assert np.array_equal(a.adjacency, b.adjacency)


def test_prune_threshold_zeros_weak_edges():
    edges = [(0, 1, 0.8), (1, 2, 0.6)]
    x, _ = _gen_sem(edges, d=3, n=2000, seed=4)
    loose = DirectLiNGAM(LingamConfig(prune_threshold=0.0)).fit(x)
    tight = DirectLiNGAM(LingamConfig(prune_threshold=0.5)).fit(x)
    assert np.count_nonzero(tight.adjacency) <= np.count_nonzero(loose.adjacency)


@pytest.mark.parametrize("bad", ["ndim", "toofew", "nlteqd", "nan"])
def test_fit_input_contract(bad):
    d = DirectLiNGAM()
    if bad == "ndim":
        with pytest.raises(ValueError):
            d.fit(np.zeros(10))
    elif bad == "toofew":
        with pytest.raises(ValueError):
            d.fit(np.zeros((10, 1)))
    elif bad == "nlteqd":
        with pytest.raises(ValueError):
            d.fit(np.zeros((3, 4)))
    elif bad == "nan":
        x = np.ones((50, 3))
        x[0, 0] = np.nan
        with pytest.raises(ValueError):
            d.fit(x)


def test_names_length_mismatch_raises():
    x, _ = _gen_sem([(0, 1, 0.8)], d=2, n=100, seed=0)
    with pytest.raises(ValueError):
        DirectLiNGAM().fit(x, ["only_one"])


def test_default_measure_is_independence_measure():
    assert isinstance(PairwiseEntropyMeasure(), IndependenceMeasure)
    # sign convention: for x -> y (y = 0.9 x + e), diff(x, y) >= 0 (x is the more exogenous)
    rng = np.random.default_rng(0)
    x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=5000)
    y = 0.9 * x + rng.uniform(-np.sqrt(3), np.sqrt(3), size=5000)
    from hymeko_rl.eval.causal.lingam import _standardize
    d = PairwiseEntropyMeasure().diff(_standardize(x), _standardize(y))
    assert d >= 0.0
