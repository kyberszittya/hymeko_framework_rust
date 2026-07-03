"""Tests for the StructuralActor — walks/cycles straight to the readout (no message-passing).

The anchor is the holonomy oracle: the precomputed ``Bᴸ`` operator equals the signed L-hop adjacency power, so
the one-matmul forward is mathematically the walk-holonomy gather.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.agents.structural_actor import StructuralActor
from hymeko_rl.experiments.structural_probe import _standardised_split, build_chain_graph, train_eval


def _raw_signed_adj(hg) -> np.ndarray:
    n = hg.n_vertices
    b = np.zeros((n, n), np.float32)
    for (u, v), s in zip(hg.edges.tolist(), hg.signs.tolist()):
        b[int(u), int(v)] = float(s)
    return b


def test_bl_reproduces_signed_two_hop() -> None:
    """THE anchor: with all length-2 walks enumerated, the precomputed ``bl`` equals the raw signed adjacency
    squared ``B²`` — so the holonomy actor's one-matmul forward IS the signed 2-hop transport."""
    hg = build_chain_graph(9, seed=0)
    actor = StructuralActor(hg, 1, 8, walk_len=2, keep=10_000)   # keep huge → every length-2 walk
    b = _raw_signed_adj(hg)
    assert np.allclose(actor.bl.numpy(), b @ b, atol=1e-5)


def test_forward_finite_and_pooled_shape() -> None:
    actor = StructuralActor(build_chain_graph(9, seed=0), 1, 8)
    out = actor(torch.randn(4, 9, 1))
    assert out.shape == (4,) and bool(torch.isfinite(out).all())


def test_agg_must_be_valid() -> None:
    with pytest.raises(ValueError, match="holonomy.*sum"):
        StructuralActor(build_chain_graph(9, seed=0), 1, 8, agg="bad")


def test_holonomy_beats_sum_on_structural_target() -> None:
    """The product-transport (holonomy) fits the structural target; the signed *sum* (v1) cannot — the ~95×
    gap the design hinged on."""
    hg = build_chain_graph(9, seed=0)
    split = _standardised_split(hg, "structural", n_train=256, n_test=512, seed=1000)
    torch.manual_seed(0)
    mse_holo = train_eval(StructuralActor(hg, 1, 24, agg="holonomy"), split, epochs=120, seed=0)
    torch.manual_seed(0)
    mse_sum = train_eval(StructuralActor(hg, 1, 24, agg="sum"), split, epochs=120, seed=0)
    assert mse_holo < 0.05, mse_holo                 # holonomy fits the 2-hop target
    assert mse_holo < 0.3 * mse_sum, (mse_holo, mse_sum)   # and far beats the signed-sum ablation
