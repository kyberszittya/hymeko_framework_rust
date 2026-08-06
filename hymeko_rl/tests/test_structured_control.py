"""Tests for the structured-control leg (Phase 2). The oracle (full-mask → CARE optimum) is the anchor."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hymeko_rl.control.structured_control import (
    lqr_cost, make_plant, mask_from_topology, run_structured_map, structured_lqr, unconstrained_lqr,
)
from hymeko_rl.control.topology_zoo import TOPOLOGIES

_N = 9
_Q = np.eye(_N)
_R = np.eye(_N)


def _plant(name: str, eps: float = 0.95):
    return make_plant(TOPOLOGIES[name](_N, seed=0), a=1.0, eps=eps)


def test_make_plant_is_hurwitz() -> None:
    """The plant is open-loop stable (so K=0 stabilises every topology)."""
    for name in ("chain", "complete", "random"):
        big_a, big_b = _plant(name)
        assert float(np.linalg.eigvals(big_a).real.max()) < 0.0
        assert big_b.shape == (_N, _N)


@pytest.mark.parametrize("plant", ["chain", "star", "grid", "complete"])
def test_oracle_full_mask_reaches_care_optimum(plant: str) -> None:
    """THE anchor: structured LQR with the full mask must converge to scipy's CARE optimum J* — validates the
    gradient + Lyapunov plumbing. (Would have failed before the warm-start fix near the stability boundary.)"""
    big_a, big_b = _plant(plant)
    _, j_star = unconstrained_lqr(big_a, big_b, _Q, _R)
    _, j_full, _ = structured_lqr(big_a, big_b, np.ones((_N, _N)), _Q, _R, iters=1000)
    assert j_full == pytest.approx(j_star, rel=1e-3), (j_full, j_star)


def test_complete_topology_is_optimal() -> None:
    """Regression for the convergence bug: the complete topology (full K) must read ρ=1.0 exactly — it can
    always achieve the unconstrained optimum. Before the warm-start it spuriously read 1.5–11.9 at strong
    coupling (sparse beating complete is impossible — it was a solver failure, not a finding)."""
    big_a, big_b = _plant("chain", eps=0.98)
    _, j_star = unconstrained_lqr(big_a, big_b, _Q, _R)
    mask = mask_from_topology(TOPOLOGIES["complete"](_N, seed=0))
    _, j_complete, _ = structured_lqr(big_a, big_b, mask, _Q, _R, iters=1000)
    assert j_complete / j_star == pytest.approx(1.0, abs=2e-3)


def test_monotonicity_denser_never_worse() -> None:
    """A denser mask is less constrained, so its cost is never higher (a structured-LQR invariant)."""
    big_a, big_b = _plant("grid")
    j_chain = structured_lqr(big_a, big_b, mask_from_topology(TOPOLOGIES["chain"](_N, seed=0)), _Q, _R)[1]
    j_grid = structured_lqr(big_a, big_b, mask_from_topology(TOPOLOGIES["grid"](_N, seed=0)), _Q, _R)[1]
    assert j_chain >= j_grid - 1e-6


def test_lqr_cost_infinite_for_unstable_closed_loop() -> None:
    """A destabilising gain yields infinite cost (the stability gate, not a silent NaN)."""
    big_a, big_b = _plant("chain")
    bad_k = -10.0 * np.eye(_N)            # u = +10 x → A − B·(−10I) = A + 10I, unstable
    assert math.isinf(lqr_cost(big_a, big_b, bad_k, _Q, _R))


def test_mask_has_diagonal_and_edges() -> None:
    hg = TOPOLOGIES["star"](_N, seed=0)
    mask = mask_from_topology(hg)
    assert np.all(np.diag(mask) == 1.0)                       # diagonal always allowed
    for i, j in hg.edges:
        assert mask[int(i), int(j)] == 1.0                   # every coupling present
    assert mask.sum() < _N * _N                               # star is sparse, not full


def test_matched_topology_minimises_suboptimality() -> None:
    """The core Phase-2 claim (reliable regime): the matched controller is the best sparse topology for each
    plant — diagonal dominance, though with small margins (benign fully-actuated plants)."""
    r = run_structured_map(names=["chain", "star", "grid", "complete"], n_nodes=_N, eps=0.95, iters=800)
    assert r["matched_is_best"]["star"], r["rho"]["star"]
    assert r["matched_is_best"]["grid"], r["rho"]["grid"]
    # and the topology effect is small here (control of a benign plant is weakly topology-dependent)
    assert max(r["worst_penalty"].values()) < 0.5
