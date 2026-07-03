"""Stage 3: structural (star-expansion) entropy H★ — the boundary behaviours and differentiability.

These pin the *signal* (uniform→1, one-hot→0, differentiable, in [0,1]) so the later exploration-seat A/B is
testing a correct quantity. They are toy-scale by design; the architecture verdict is a separate real-topology
run (standing rule)."""
from __future__ import annotations

import math

import numpy as np
import torch

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.agents.policy import HSiKANBackbone
from hymeko_rl.agents.star_entropy import mean_star_entropy, star_expansion_entropy


def test_uniform_activation_is_max_entropy() -> None:
    """Equal per-vertex energy → H★ = 1 (fully diffuse reasoning over the structure)."""
    act = torch.ones(2, 5, 3)                       # every vertex same norm → uniform energy
    h = star_expansion_entropy(act)
    assert h.shape == (2,)
    assert torch.allclose(h, torch.ones(2), atol=1e-5)


def test_one_hot_activation_is_zero_entropy() -> None:
    """All energy on one vertex → H★ = 0 (maximally concentrated)."""
    act = torch.zeros(1, 4, 3)
    act[0, 2, :] = 1.0                              # only vertex 2 carries energy
    h = star_expansion_entropy(act)
    assert torch.allclose(h, torch.zeros(1), atol=1e-5)


def test_entropy_normalized_range_and_unnormalized_nats() -> None:
    """Normalized H★ ∈ [0,1]; unnormalized peaks at log N nats for the uniform case."""
    act = torch.ones(1, 8, 2)
    assert torch.allclose(star_expansion_entropy(act, normalize=False), torch.tensor([math.log(8)]), atol=1e-5)
    assert 0.0 <= float(star_expansion_entropy(act)[0]) <= 1.0 + 1e-6


def test_entropy_is_differentiable() -> None:
    """H★ has a gradient w.r.t. the activations — it can augment the RL objective (the seat use)."""
    act = torch.randn(3, 6, 4, requires_grad=True)
    star_expansion_entropy(act).sum().backward()
    assert act.grad is not None and torch.isfinite(act.grad).all()


def test_rank_guard() -> None:
    try:
        star_expansion_entropy(torch.randn(5, 3))   # rank-2, not (B, N, H)
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-rank-3 input")


def test_mean_star_entropy_from_hsikan_backbone() -> None:
    """End-to-end: H★ read straight from an HSiKAN backbone's per-vertex activations is a finite scalar in
    [0,1] and differentiable in the backbone params (so the seat term can train)."""
    hg = HypergraphState(vertex_labels=("a", "b", "c"),
                         edges=np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64),
                         signs=np.array([1, -1, 1, -1], dtype=np.int64), topo_hash="test")
    backbone = HSiKANBackbone(in_feat=2, hidden=4, n_layers=2, hg_state=hg)
    obs = torch.randn(2, 3, 2)
    h = mean_star_entropy(backbone, obs)
    assert h.dim() == 0 and 0.0 <= float(h.detach()) <= 1.0 + 1e-6
    h.backward()
    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
