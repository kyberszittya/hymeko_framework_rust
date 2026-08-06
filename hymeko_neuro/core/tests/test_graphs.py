"""Tests for the shared signed-adjacency builder (engine-wise reuse; the OTC harness now uses it)."""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.core import DenseBatchedBackend, SparseSignedBackend, build_signed_adjacency


def test_signs_separated_and_no_leakage() -> None:
    # + edges (0,1),(1,2); - edge (0,2); node 3 isolated.
    a_pos, a_neg = build_signed_adjacency([[0, 1], [1, 2], [0, 2]], [1, 1, -1], 4)
    dp, dn = a_pos.to_dense(), a_neg.to_dense()
    assert a_pos.is_sparse and a_neg.is_sparse and dp.shape == (4, 4)
    assert dn[0, 2] > 0 and dp[0, 2] == 0          # the - edge lives only in a_neg
    assert dp[0, 1] > 0 and dn[0, 1] == 0          # the + edge lives only in a_pos
    assert dp[3].sum() == 0 and dn[3].sum() == 0   # isolated node: nothing leaks in


def test_symmetry_prenorm_and_rownorm() -> None:
    # symmetric before row-norm (undirected); row-norm then makes each non-empty row sum to 1 (degree-normalised,
    # which intentionally breaks symmetry — receiver-side normalisation).
    a_raw, _ = build_signed_adjacency([[0, 1], [1, 2]], [1, 1], 3, row_normalize=False)
    d = a_raw.to_dense()
    assert torch.allclose(d, d.T)
    a_norm, _ = build_signed_adjacency([[0, 1], [1, 2]], [1, 1], 3)
    r = a_norm.to_dense().sum(dim=1)
    assert torch.allclose(r[r > 0], torch.ones_like(r[r > 0]))


def test_dense_sparse_parity_from_builder() -> None:
    a_pos, a_neg = build_signed_adjacency([[0, 1], [1, 2]], [1, -1], 3)
    h = torch.randn(1, 3, 4)
    sp = SparseSignedBackend().aggregate(a_pos, a_neg, h)
    dp = DenseBatchedBackend().aggregate(a_pos.to_dense(), a_neg.to_dense(), h)
    assert torch.allclose(sp[0], dp[0], atol=1e-6) and torch.allclose(sp[1], dp[1], atol=1e-6)


def test_errors() -> None:
    with pytest.raises(ValueError, match="E, 2"):
        build_signed_adjacency([0, 1], [1], 3)
