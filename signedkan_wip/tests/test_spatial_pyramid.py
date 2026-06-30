"""Tests for the reusable SpatialPyramidPool (compressed flatten readout)."""
from __future__ import annotations

import torch

from signedkan_wip.src.vision.spatial_pyramid import (
    SpatialPyramidPool,
    grid_positions,
    pooling_matrix,
)


def _scatter_reference(features, positions, levels, d):
    """Naive scatter-mean pyramid pool (pre-matmul implementation) for parity."""
    cells = []
    for lvl in levels:
        idx = (positions * lvl).floor().clamp(0, lvl - 1).long()
        flat = idx[:, 0] * lvl + idx[:, 1]
        summed = features.new_zeros(lvl * lvl, d).index_add(0, flat, features)
        count = features.new_zeros(lvl * lvl).index_add(
            0, flat, torch.ones_like(flat, dtype=features.dtype))
        cells.append(summed / count.clamp_min(1.0).unsqueeze(-1))
    return torch.cat(cells, dim=0).reshape(-1)


def test_out_dim_is_item_count_independent() -> None:
    pool = SpatialPyramidPool(8, levels=(1, 2, 4))
    assert pool.out_dim == (1 + 4 + 16) * 8
    for n in (1, 7, 200):                       # variable item counts
        out = pool(torch.randn(n, 8), torch.rand(n, 2))
        assert out.shape == (pool.out_dim,) and torch.isfinite(out).all()


def test_empty_cells_are_zero_not_nan() -> None:
    """One item → only its cells are non-zero across levels; the rest are 0."""
    pool = SpatialPyramidPool(4, levels=(2,))    # 4 cells
    out = pool(torch.ones(1, 4), torch.tensor([[0.1, 0.1]]))  # top-left cell
    cells = out.view(4, 4)
    assert torch.allclose(cells[0], torch.ones(4))
    assert torch.allclose(cells[1:], torch.zeros(3, 4))


def test_position_changes_the_descriptor() -> None:
    pool = SpatialPyramidPool(4, levels=(1, 2, 4))
    feat = torch.randn(5, 4, generator=torch.Generator().manual_seed(0))
    p1 = torch.rand(5, 2, generator=torch.Generator().manual_seed(1))
    p2 = torch.rand(5, 2, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(pool(feat, p1), pool(feat, p2))


def test_grid_positions_match_binning_on_even_grid() -> None:
    """On a 12-grid binned to 4×4 the pyramid pool equals an exact 3×3-block
    mean (binning ≡ grid pooling when the grid divides evenly)."""
    h = w = 12
    pos = grid_positions(h, w)
    assert pos.shape == (144, 2)
    feat = torch.randn(144, 1)
    pool = SpatialPyramidPool(1, levels=(4,))
    out = pool(feat, pos).view(16)
    block = feat.view(12, 12).unfold(0, 3, 3).unfold(1, 3, 3).mean(dim=(-1, -2))
    assert torch.allclose(out, block.reshape(-1), atol=1e-5)


def test_gate_default_is_off() -> None:
    """Default is the gate-free pyramid (the gate ablation found the gate inert)."""
    assert SpatialPyramidPool(8).gate is None
    assert SpatialPyramidPool(8, dynamic=True).gate is not None


def test_matmul_form_matches_scatter_reference() -> None:
    """The P@features matmul equals the naive scatter-mean pyramid (the GPU
    optimisation is numerically faithful)."""
    pool = SpatialPyramidPool(8, levels=(1, 2, 4))
    feat = torch.randn(50, 8, generator=torch.Generator().manual_seed(0))
    pos = torch.rand(50, 2, generator=torch.Generator().manual_seed(1))
    ref = _scatter_reference(feat, pos, (1, 2, 4), 8)
    assert torch.allclose(pool(feat, pos), ref, atol=1e-5)


def test_batched_equals_per_item_loop() -> None:
    """One batched matmul == looping the pool per item (the batching is exact)."""
    pool = SpatialPyramidPool(6, levels=(1, 2))
    pos = torch.rand(40, 2, generator=torch.Generator().manual_seed(2))
    pool.set_fixed_positions(pos)
    feats = torch.randn(8, 40, 6, generator=torch.Generator().manual_seed(3))
    batched = pool(feats)                                   # (8, out_dim)
    looped = torch.stack([pool(feats[b]) for b in range(8)], dim=0)
    assert batched.shape == (8, pool.out_dim)
    assert torch.allclose(batched, looped, atol=1e-6)


def test_fixed_P_matches_per_call_positions() -> None:
    pool = SpatialPyramidPool(4, levels=(1, 2, 4))
    pos = grid_positions(12, 12)
    feat = torch.randn(144, 4, generator=torch.Generator().manual_seed(4))
    pool.set_fixed_positions(pos)
    assert torch.allclose(pool(feat), pool(feat, pos), atol=1e-6)


def test_pooling_matrix_rows_are_means() -> None:
    """Each non-empty P row sums to 1 (a mean); empty rows sum to 0."""
    p = pooling_matrix(grid_positions(8, 8), (1, 2, 4))
    sums = p.sum(dim=1)
    assert torch.all((torch.isclose(sums, torch.ones_like(sums)))
                     | torch.isclose(sums, torch.zeros_like(sums)))
