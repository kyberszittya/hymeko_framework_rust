"""Tests for structured MPC (Phase 2b). The oracle (unconstrained matched MPC = discrete-LQR) is the anchor."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from hymeko_rl.structured_control import make_plant, mask_from_topology, unconstrained_lqr
from hymeko_rl.structured_mpc import (
    SaturatedLQR, StructuredMPC, discretize, run_mpc_topology_map, simulate_closed_loop,
)
from hymeko_rl.topology_zoo import TOPOLOGIES

_N = 9
_Q = np.eye(_N)
_R = np.eye(_N)
_X0 = 3.0 * np.ones(_N)


def _discrete(name: str, eps: float = 0.9):
    a_c, b_c = make_plant(TOPOLOGIES[name](_N, seed=0), eps=eps)
    return (*discretize(a_c, b_c, 0.1), a_c, b_c)


def test_discretize_preserves_stability() -> None:
    """A Hurwitz continuous plant discretises to a Schur (spectral radius < 1) one."""
    a_d, _, _, _ = _discrete("chain")
    assert float(np.abs(np.linalg.eigvals(a_d)).max()) < 1.0


@pytest.mark.parametrize("plant", ["chain", "star", "grid"])
def test_oracle_unconstrained_matched_mpc_equals_discrete_lqr(plant: str) -> None:
    """THE anchor: with no input bound and the matched model, the MPC closed-loop cost equals the discrete-LQR
    cost-to-go x₀ᵀP_d x₀ — validates the condensed prediction + terminal cost."""
    a_d, b_d, _, _ = _discrete(plant)
    j_d = float(_X0 @ solve_discrete_are(a_d, b_d, _Q, _R) @ _X0)
    mpc = StructuredMPC(a_d, b_d, _Q, _R, horizon=12, u_max=math.inf)
    cost = simulate_closed_loop(a_d, b_d, mpc, _X0, steps=200, q=_Q, r=_R, u_max=math.inf)
    assert cost == pytest.approx(j_d, rel=1e-4)


def test_constrained_mpc_beats_saturated_lqr() -> None:
    """MPC's reason to exist: under saturation it must do no worse than clipping the LQR gain."""
    a_d, b_d, a_c, b_c = _discrete("chain")
    mpc = StructuredMPC(a_d, b_d, _Q, _R, horizon=12, u_max=0.6)
    c_mpc = simulate_closed_loop(a_d, b_d, mpc, _X0, steps=60, q=_Q, r=_R, u_max=0.6)
    k_star, _ = unconstrained_lqr(a_c, b_c, _Q, _R)
    k_m = k_star * mask_from_topology(TOPOLOGIES["chain"](_N, seed=0))
    c_sat = simulate_closed_loop(a_d, b_d, SaturatedLQR(k_m), _X0, steps=60, q=_Q, r=_R, u_max=0.6)
    assert c_mpc <= c_sat + 1e-6


def test_mpc_respects_input_box() -> None:
    a_d, b_d, _, _ = _discrete("star")
    mpc = StructuredMPC(a_d, b_d, _Q, _R, horizon=10, u_max=0.6)
    u = mpc.control(_X0)
    assert np.all(np.abs(u) <= 0.6 + 1e-6)


def test_matched_model_is_best_under_constrained_mpc() -> None:
    """The Phase-2b claim: with input saturation, the MPC whose model matches the plant controls best — a clean
    diagonal (model mismatch compounds over the horizon)."""
    r = run_mpc_topology_map(names=["chain", "star", "grid", "complete"], n_nodes=_N, u_max=0.6, steps=50)
    assert all(r["matched_is_best"].values()), r["ratio"]
