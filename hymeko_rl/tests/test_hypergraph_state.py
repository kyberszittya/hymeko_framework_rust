"""The kinematic-hypergraph bridge + the HSiKAN policy reading it.

Run: pytest -p no:randomly hymeko_rl/tests/test_hypergraph_state.py
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.agents.policy import ActorCritic, build_policy

_REPO = Path(__file__).resolve().parents[2]
# The hero-emitted faithful arm MJCF (compiled from data/robotics/anthropomorphic_arm.hymeko).
_ARM_MJCF = _REPO / "demos" / "hero" / "out" / "anthropomorphic_arm_faithful.mjcf"


def _arm_hg() -> HypergraphState:
    if not _ARM_MJCF.exists():
        pytest.skip(f"emitted arm MJCF not present: {_ARM_MJCF}")
    return HypergraphState.from_mjcf(_ARM_MJCF)


def test_bridge_extracts_links_and_signed_joints() -> None:
    hg = _arm_hg()
    assert hg.n_vertices >= 4                       # links (world dropped)
    # each joint -> a down (+1) and an up (-1) arc; balanced count.
    assert (hg.signs == 1).sum() == (hg.signs == -1).sum() > 0
    assert hg.edges.shape == (len(hg.signs), 2)
    assert hg.topo_hash and hg.edges.max() < hg.n_vertices


def test_dense_signed_adj_is_row_normalised_and_shaped() -> None:
    hg = _arm_hg()
    a_pos, a_neg = hg.dense_signed_adj()
    n = hg.n_vertices
    assert a_pos.shape == (n, n) and a_neg.shape == (n, n)
    # row sums are 0 (isolated row) or 1 (normalised) — never > 1.
    for a in (a_pos, a_neg):
        rs = a.sum(dim=1)
        assert torch.all(rs <= 1.0 + 1e-6) and torch.all(rs >= -1e-6)


def test_topo_hash_is_stable_across_reloads() -> None:
    assert _arm_hg().topo_hash == _arm_hg().topo_hash


def test_hsikan_policy_reads_the_hypergraph_not_raw_joints() -> None:
    """build_policy('hsikan', hg_state=...) constructs an actor-critic whose backbone
    consumes per-vertex features on the kinematic graph (B, N, in_feat) — the distinctive
    claim, end-to-end at the network level."""
    torch.manual_seed(0)
    hg = _arm_hg()
    n, in_feat = hg.n_vertices, 4
    ac = build_policy("hsikan", obs_dim=in_feat, action_dim=hg.n_vertices,
                      hg_state=hg, hidden=16)
    assert isinstance(ac, ActorCritic) and ac.n_parameters() > 0
    obs = torch.randn(5, n, in_feat)                # per-vertex dynamic state
    action, log_prob, value = ac.act(obs)
    assert action.shape == (5, n) and log_prob.shape == (5,) and value.shape == (5,)
    lp, ent, val = ac.evaluate(obs, action)
    assert lp.requires_grad and torch.isfinite(ent).all()  # diff. entropy may be < 0


def test_hsikan_backbone_rejects_wrong_node_count() -> None:
    hg = _arm_hg()
    ac = build_policy("hsikan", obs_dim=3, action_dim=2, hg_state=hg, hidden=8)
    with pytest.raises(ValueError):
        ac.act(torch.randn(2, hg.n_vertices + 1, 3))   # wrong N
