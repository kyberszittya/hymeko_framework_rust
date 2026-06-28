"""Hyperedge-augmented graph: put the object + goal in the graph (grasp/goal hyperedges) — the fix for the
Galambos HSiKAN==MLP tie (the coin/zone were absent from the structure). Graph construction only; no training."""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.hypergraph_state import HypergraphState
from hymeko_rl.policy import HSiKANBackbone


def _galambos_arms() -> HypergraphState:
    """A synthetic two-arm Galambos graph: 6 robot vertices, two 2-link chains (no mujoco)."""
    return HypergraphState(
        vertex_labels=("base_left", "link1_left", "link2_left", "base_right", "link1_right", "link2_right"),
        edges=np.array([[0, 1], [1, 0], [1, 2], [2, 1], [3, 4], [4, 3], [4, 5], [5, 4]], dtype=np.int64),
        signs=np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64), topo_hash="galambos")


def _members_of(hg: HypergraphState, hub: int) -> set[int]:
    return {int(s if d == hub else d) for (s, d) in hg.edges if hub in (int(s), int(d))}


def test_augmented_vertex_count_and_labels() -> None:
    """6 robot + {coin, zone} entities + {grasp, goal} hubs = 10 vertices; labels in order; re-encode gate."""
    arm = _galambos_arms()
    tip_l, tip_r, coin, zone = 2, 5, 6, 7
    aug = arm.with_task_hyperedges(
        new_entities=["coin", "zone"],
        hyperedges=[("grasp_hub", [tip_l, tip_r, coin]), ("goal_hub", [coin, zone])])
    assert aug.n_vertices == 10
    assert aug.vertex_labels[6:] == ("coin", "zone", "grasp_hub", "goal_hub")
    assert aug.topo_hash.endswith(":task")


def test_grasp_and_goal_hub_connectivity() -> None:
    """The grasp hub joins both fingertips + the coin; the goal hub joins the coin + the zone."""
    arm = _galambos_arms()
    tip_l, tip_r, coin, zone = 2, 5, 6, 7
    aug = arm.with_task_hyperedges(
        new_entities=["coin", "zone"],
        hyperedges=[("grasp_hub", [tip_l, tip_r, coin]), ("goal_hub", [coin, zone])])
    grasp_hub, goal_hub = 8, 9
    assert _members_of(aug, grasp_hub) == {tip_l, tip_r, coin}
    assert _members_of(aug, goal_hub) == {coin, zone}


def test_signed_incidence_member_into_hub_out() -> None:
    """member→hub is +1 (a_pos), hub→member is -1 (a_neg) — consistent with star_expansion's signing."""
    aug = _galambos_arms().with_task_hyperedges(
        new_entities=["coin"], hyperedges=[("grasp_hub", [2, 5, 6])])
    grasp_hub = 7
    pos = {(int(s), int(d)) for (s, d), sg in zip(aug.edges, aug.signs) if sg > 0 and grasp_hub in (int(s), int(d))}
    neg = {(int(s), int(d)) for (s, d), sg in zip(aug.edges, aug.signs) if sg < 0 and grasp_hub in (int(s), int(d))}
    assert (2, grasp_hub) in pos and (grasp_hub, 2) in neg


def test_feeds_hsikan_backbone() -> None:
    """The augmented graph is a drop-in for the HSiKAN backbone; activations span robot+entities+hubs."""
    aug = _galambos_arms().with_task_hyperedges(
        new_entities=["coin", "zone"],
        hyperedges=[("grasp_hub", [2, 5, 6]), ("goal_hub", [6, 7])])
    a_pos, a_neg = aug.dense_signed_adj()
    assert a_pos.shape == (10, 10)
    backbone = HSiKANBackbone(in_feat=3, hidden=8, n_layers=2, hg_state=aug)
    acts = backbone.node_activations(torch.randn(4, 10, 3))
    assert acts.shape == (4, 10, 8)


def test_hyperedge_on_hyperedge() -> None:
    """A later hyperedge may reference an earlier hub (edge-on-edge incidence) — bundle-of-bundles."""
    aug = _galambos_arms().with_task_hyperedges(
        new_entities=["coin", "zone"],
        hyperedges=[("grasp_hub", [2, 5, 6]),          # hub at index 8
                    ("meta_hub", [8, 7])])             # references the grasp hub (8) + zone (7)
    assert aug.n_vertices == 10                         # 6 robot + {coin,zone} + {grasp_hub, meta_hub}
    assert _members_of(aug, 9) == {8, 7}               # meta_hub (idx 9) joins grasp_hub (8) + zone (7)


def test_out_of_range_member_raises() -> None:
    arm = _galambos_arms()
    try:
        arm.with_task_hyperedges(new_entities=["coin"], hyperedges=[("grasp_hub", [2, 99])])
    except ValueError:
        return
    raise AssertionError("expected ValueError on an out-of-range member")
