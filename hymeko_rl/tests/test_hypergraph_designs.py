"""Tests for the combinatorial-design hypergraph generators (Kato's Steiner/k-uniform/sunflower extension)."""
from __future__ import annotations

import itertools

import pytest
import torch

from hymeko_rl.hypergraph_designs import (
    DESIGNS, design_to_state, k_uniform_random, steiner_triple_system, sunflower,
)
from hymeko_rl.structural_probe import build_model


def _blocks(hg, n_base: int) -> list[tuple[int, ...]]:
    """Recover each block as the base members (vertex < n_base) incident to its hub (vertex >= n_base)."""
    out = []
    for hub in range(n_base, hg.n_vertices):
        out.append(tuple(sorted({int(a) for a, b in hg.edges if int(b) == hub and int(a) < n_base})))
    return out


@pytest.mark.parametrize(("n", "n_blocks"), [(7, 7), (9, 12)])
def test_steiner_every_pair_covered_exactly_once(n: int, n_blocks: int) -> None:
    """The defining S(2,3,n) property: every pair of base vertices lies on exactly one triple."""
    hg = steiner_triple_system(n)
    blocks = _blocks(hg, n)
    assert len(blocks) == n_blocks
    assert all(len(b) == 3 for b in blocks)
    pair_count: dict[tuple[int, int], int] = {}
    for b in blocks:
        for p in itertools.combinations(b, 2):
            pair_count[p] = pair_count.get(p, 0) + 1
    assert all(pair_count.get(p, 0) == 1 for p in itertools.combinations(range(n), 2))


def test_steiner_rejects_unsupported_n() -> None:
    with pytest.raises(ValueError, match="STS implemented"):
        steiner_triple_system(8)


def test_k_uniform_blocks_are_distinct_k_subsets() -> None:
    hg = k_uniform_random(9, 4, n_blocks=8, seed=0)
    blocks = _blocks(hg, 9)
    assert len(blocks) == 8
    assert all(len(b) == 4 for b in blocks)
    assert len(set(blocks)) == 8                               # distinct


def test_sunflower_blocks_intersect_exactly_in_core() -> None:
    core_size, petal, petals = 2, 2, 3
    hg = sunflower(core_size=core_size, petal_size=petal, n_petals=petals)
    blocks = _blocks(hg, core_size + petal * petals)
    core = set(range(core_size))
    assert len(blocks) == petals
    for a, b in itertools.combinations(blocks, 2):
        assert set(a) & set(b) == core                         # pairwise intersection = the core
    assert all(set(core).issubset(b) and len(b) == core_size + petal for b in blocks)


def test_design_to_state_rejects_bad_blocks() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        design_to_state(5, [(0, 1, 9)], tag="bad")
    with pytest.raises(ValueError, match="empty"):
        design_to_state(5, [()], tag="bad")


@pytest.mark.parametrize("name", list(DESIGNS))
def test_designs_instantiate_as_hsikan_controllers(name: str) -> None:
    """Every design star-expands to a valid signed graph an HSiKAN controller runs on (finite forward) — the
    bridge from combinatorial design to controller."""
    hg = DESIGNS[name]()
    a_pos, a_neg = hg.dense_signed_adj()
    assert a_pos.shape[0] == hg.n_vertices and bool(torch.isfinite(a_pos).all())
    model = build_model("hsikan", hg, 16, n_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(3, hg.n_vertices, 1))
    assert out.shape[0] == 3 and bool(torch.isfinite(out).all())
