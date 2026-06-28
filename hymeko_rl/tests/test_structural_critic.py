"""Structural critic: motif enumeration (cycles+walks), the aggregation×pooling ablation grid (forward shape +
gradient), per-motif decomposition, config validation. Requires the built `hymeko` binding."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.hypergraph_state import HypergraphState
from hymeko_rl.policy import hsikan_backbone
from hymeko_rl.structural_critic import (
    AGGREGATIONS,
    POOLINGS,
    StructuralCritic,
    StructuralCriticConfig,
    enumerate_motifs,
)


def _hg() -> HypergraphState:
    return HypergraphState(
        vertex_labels=("a", "b", "c", "d"),
        edges=np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64),
        signs=np.array([1, -1, 1, -1], dtype=np.int64),
        topo_hash="test")


def _backbone(hg: HypergraphState, d: int = 16) -> tuple[object, int]:
    return hsikan_backbone(4, hg_state=hg, hidden=d, n_layers=2, skip="highway")


def test_enumerate_motifs_cycles_and_walks() -> None:
    vid, sgn, length = enumerate_motifs(_hg(), StructuralCriticConfig())
    assert vid.shape[0] > 0 and vid.shape == sgn.shape and length.shape[0] == vid.shape[0]
    assert (length >= 1).all()          # the triangle 0-1-2 + walks are found


@pytest.mark.parametrize("agg", AGGREGATIONS)
@pytest.mark.parametrize("pool", POOLINGS)
def test_forward_shape_and_grad(agg: str, pool: str) -> None:
    torch.manual_seed(0)
    hg = _hg()
    bb, feat = _backbone(hg)
    sc = StructuralCritic(bb, feat, hg, StructuralCriticConfig(aggregation=agg, pooling=pool))
    obs = torch.randn(8, 4, 4)
    v = sc.value(obs)
    assert v.shape == (8,)
    pmv = sc.per_motif_values(obs)
    assert pmv.shape == (8, sc.n_motifs) and sc.n_motifs > 0
    v.pow(2).mean().backward()
    assert sc.value_head.weight.grad is not None and sc.value_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in sc.backbone.parameters())


def test_config_validation() -> None:
    hg = _hg()
    bb, feat = _backbone(hg)
    with pytest.raises(ValueError):
        StructuralCritic(bb, feat, hg, StructuralCriticConfig(aggregation="bad"))
    with pytest.raises(ValueError):
        StructuralCritic(bb, feat, hg, StructuralCriticConfig(use_cycles=False, use_walks=False))
