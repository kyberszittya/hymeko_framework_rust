"""The signed-graph convention of the general HSiKAN: same CR body, edge-sign head, sparse-or-dense backend.

Proves the "one HSiKAN, changeable head" claim — the signed-graph model reuses :class:`SignedKANBackbone`
(the same body the RL line uses) and only swaps the input adapter (embeddings) + head (edge-sign), and that the
sparse backend agrees with the dense one so it scales to large signed graphs (e.g. OTC) on the CR spline.
"""
from __future__ import annotations

import pytest
import torch

from signed_kan import (
    DenseBatchedBackend,
    EdgeSignHead,
    SignedGraphHSiKAN,
    SparseSignedBackend,
)

# A tiny signed graph: 4 nodes, a couple of + and - edges.
_A_POS = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
_A_NEG = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


# ── edge-sign head ───────────────────────────────────────────────────────────
def test_edge_sign_head_shapes() -> None:
    head = EdgeSignHead(8, n_classes=1)
    reps = torch.randn(4, 8)
    edges = torch.tensor([[0, 1], [2, 3], [1, 3]])
    assert head(reps, edges).shape == (3, 1)
    with pytest.raises(ValueError):
        head(torch.randn(4, 8, 1), edges)          # node_reps must be 2-D
    with pytest.raises(ValueError):
        head(reps, torch.tensor([0, 1]))           # edges must be (E, 2)


# ── sparse vs dense backend parity ───────────────────────────────────────────
def test_sparse_matches_dense_backend() -> None:
    h = torch.randn(1, 4, 5)
    dp, dn = DenseBatchedBackend().aggregate(_A_POS, _A_NEG, h)
    sp, sn = SparseSignedBackend().aggregate(_A_POS, _A_NEG, h)
    assert torch.allclose(dp, sp, atol=1e-6) and torch.allclose(dn, sn, atol=1e-6)


# ── the general HSiKAN under the signed-graph convention ─────────────────────
def test_signed_graph_hsikan_forward_and_learns() -> None:
    """Same CR body as the RL line + edge-sign head: forwards to per-edge logits, and a gradient step reduces a
    sign-classification loss (so the unified body + swapped head actually trains on a signed graph)."""
    torch.manual_seed(0)
    model = SignedGraphHSiKAN(4, _A_POS, _A_NEG, hidden=8, n_layers=2)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 2]])
    targets = torch.tensor([[1.0], [1.0], [1.0], [0.0]])    # + + + -
    logits = model(edges)
    assert logits.shape == (4, 1) and torch.isfinite(logits).all()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss0 = torch.nn.functional.binary_cross_entropy_with_logits(model(edges), targets)
    for _ in range(50):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(edges), targets).backward()
        opt.step()
    loss1 = torch.nn.functional.binary_cross_entropy_with_logits(model(edges), targets)
    assert loss1.item() < loss0.item()          # the unified CR body + edge head learns


def test_signed_graph_hsikan_with_sparse_backend_and_highway() -> None:
    """Scales via the sparse backend and runs with the CR body's highway skip + weighted incidence on."""
    torch.manual_seed(0)
    model = SignedGraphHSiKAN(4, _A_POS, _A_NEG, hidden=8, n_layers=2, skip="highway",
                              incidence="weighted", backend=SparseSignedBackend())
    edges = torch.tensor([[0, 1], [2, 3]])
    out = model(edges)
    assert out.shape == (2, 1) and torch.isfinite(out).all()
