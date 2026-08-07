"""R12 / HSiKAN-1 — unit tests for the transportability critic models + incidence builders.

Guards the invariants the architecture ablation *depends on* to be fair:
  * the HSiKAN edge/update functions are SHARED across edges ⇒ param count is independent of the incidence, so A1-A3
    differ only in structure (the whole premise of the matched-budget comparison);
  * all incidence builders yield valid, in-range node indices of the intended edge sizes;
  * the (Phase-1b) stronger forward — mean‖max pooling, 2 residual rounds, concat readout — runs and is finite;
  * MatchedModels keeps every model inside a tight param band.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.coin_delivery.transportability_critic import (
    INPUT_DIM, NODE_NAMES, HypergraphNet, MatchedModels, MLPNet, build_input_row,
    count_params, degree_matched_incidence, object_features, random_sparse_incidence,
    steiner_incidence, task_incidence)

_N = len(NODE_NAMES)


def _valid_incidence(edges: list[tuple[int, ...]]) -> bool:
    return all(len(e) >= 2 and all(0 <= v < _N for v in e) and len(set(e)) == len(e) for e in edges)


def test_incidence_builders_yield_valid_indices() -> None:
    rng = np.random.default_rng(0)
    task = task_incidence()
    assert _valid_incidence(task)
    assert _valid_incidence(steiner_incidence())
    assert _valid_incidence(random_sparse_incidence(rng, len(task)))
    assert _valid_incidence(degree_matched_incidence(rng, task))


def test_degree_matched_preserves_degree_sequence_and_sizes() -> None:
    """The Steiner control must match target degree sequence + edge sizes — else it isn't a fair control."""
    rng = np.random.default_rng(1)
    task = task_incidence()
    ctrl = degree_matched_incidence(rng, task)
    assert sorted(len(e) for e in ctrl) == sorted(len(e) for e in task)
    deg = lambda edges: tuple(sorted(sum(v in e for e in edges) for v in range(_N)))  # noqa: E731
    assert deg(ctrl) == deg(task)


def test_hypergraph_params_independent_of_incidence() -> None:
    """CORE INVARIANT of the ablation: shared edge/update fns ⇒ identical param count for any incidence at fixed
    (node_dim, hidden, rounds). If this breaks, A1-A3 are no longer matched and the comparison is invalid."""
    rng = np.random.default_rng(2)
    task = task_incidence()
    counts = {
        count_params(HypergraphNet(task)),
        count_params(HypergraphNet(steiner_incidence())),
        count_params(HypergraphNet(random_sparse_incidence(rng, len(task)))),
        count_params(HypergraphNet(degree_matched_incidence(rng, task))),
        count_params(HypergraphNet([(0, 1), (2, 3, 4, 5, 6, 7, 8, 9)])),   # extreme arities, same param count
    }
    assert len(counts) == 1, f"param count varies with incidence: {counts}"


def test_stronger_forward_shape_and_finite() -> None:
    """Phase-1b forward: mean‖max + 2 residual rounds + concat readout → finite logits of shape (B,)."""
    torch.manual_seed(0)
    x = torch.randn(7, INPUT_DIM)
    for model in (MLPNet(), HypergraphNet(task_incidence())):
        out = model(x)
        assert out.shape == (7,)
        assert torch.isfinite(out).all()


def test_rounds_add_parameters() -> None:
    """A regression on the Phase-1b multi-round change: rounds are real extra layers, not a no-op."""
    assert count_params(HypergraphNet(task_incidence(), rounds=2)) > count_params(
        HypergraphNet(task_incidence(), rounds=1))


def test_matched_models_within_param_band() -> None:
    ps = [count_params(m) for m in MatchedModels().build(0).values()]
    assert max(ps) / min(ps) < 1.2, f"param band too wide: {ps}"
    assert len(MatchedModels().build(0)) == 5


def test_build_input_row_layout() -> None:
    x = list(range(30))
    theta = [0.1] * 6
    row = build_input_row(x, theta, "O0")
    assert len(row) == INPUT_DIM
    assert row[30:36] == theta
    assert row[36:41] == object_features("O0")


def test_object_features_shape_encoding() -> None:
    assert object_features("O0")[:2] == [1.0, 0.0]      # cylinder one-hot
    assert object_features("O4-S")[:2] == [0.0, 1.0]    # box one-hot
    assert object_features("O2-M")[4] == 2.0            # mass ratio


def test_unknown_family_raises() -> None:
    """Failure case: an unregistered family is a caller bug, surfaced as KeyError not a silent default."""
    try:
        object_features("O9-does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown family")


# ---- metric backbone (from the ablation harness — every reported number rides on these) -----------
def test_auroc_known_cases() -> None:
    from hymeko_rl.experiments.r12_hsikan1_ablation import _auroc
    y = np.array([1, 1, 0, 0], float)
    assert _auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0        # perfect separation
    assert _auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0        # perfectly reversed
    assert _auroc(y, np.array([0.6, 0.4, 0.5, 0.3])) == 0.75       # 3 of 4 pos>neg pairs (continuous, tie-free)
    assert np.isnan(_auroc(np.array([1.0, 1.0]), np.array([0.5, 0.6])))  # single-class ⇒ undefined


def test_top1_k6_picks_argmax_prediction() -> None:
    from hymeko_rl.experiments.r12_hsikan1_ablation import _top1_k6
    rows = [{"handoff_family": "A", "scenario": "s", "seed": 0, "k6": False},
            {"handoff_family": "A", "scenario": "s", "seed": 0, "k6": True},
            {"handoff_family": "A", "scenario": "s", "seed": 0, "k6": False}]
    idx = [0, 1, 2]
    top1, oracle = _top1_k6(rows, idx, np.array([0.1, 0.9, 0.2]))   # top-1 lands on the delivering θ
    assert (top1, oracle) == (1.0, 1.0)
    top1, oracle = _top1_k6(rows, idx, np.array([0.9, 0.1, 0.2]))   # top-1 misses; oracle still sees the positive
    assert (top1, oracle) == (0.0, 1.0)
