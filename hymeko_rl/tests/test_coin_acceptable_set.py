"""Acceptable-set multimodality discriminating test — pure clustering / assignment / verdict logic (fast, no physics).

These functions decide whether the next fix is a K-head multimodal proposal (mode structure exists) or a representation
change (one connected acceptable blob). The clustering must be single-linkage connected-components; the verdict must call
MULTIMODAL only on genuine separation (inter-basin > intra-basin) or a demonstrably non-delivering centroid; and the
held-out overlay must flag an orphan (OOD) delivering θ.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.acceptable_set import (
    assign_to_basins, cluster_basins, multimodality_verdict)


# ───────────────────────────── clustering (single-linkage connected components) ─────────────────────────────
def test_cluster_two_separated_blobs_gives_two_basins():
    a = np.random.default_rng(0).normal(0, 0.05, (8, 6)) + np.array([0, 0, 0, 0, 0, 0.0])
    b = np.random.default_rng(1).normal(0, 0.05, (8, 6)) + np.array([3.0, 0, 0, 0, 0, 0])
    cl = cluster_basins(np.vstack([a, b]), link_tol=0.5)
    assert cl["n_basins"] == 2
    assert cl["min_inter_basin_dist"] > cl["max_intra_nn_hop"]      # separated: gap > tightness
    assert sorted(cl["basin_sizes"]) == [8, 8]


def test_cluster_one_blob_is_single_basin():
    x = np.random.default_rng(2).normal(0, 0.05, (16, 6))
    cl = cluster_basins(x, link_tol=0.5)
    assert cl["n_basins"] == 1
    assert cl["min_inter_basin_dist"] is None                       # no inter-basin distance for one basin


def test_cluster_link_tol_merges_at_larger_threshold():
    a = np.zeros((4, 6))
    b = np.zeros((4, 6)) + np.array([1.0, 0, 0, 0, 0, 0])           # blobs 1.0 apart
    X = np.vstack([a, b])
    assert cluster_basins(X, link_tol=0.5)["n_basins"] == 2          # gap 1.0 > tol 0.5 → split
    assert cluster_basins(X, link_tol=1.5)["n_basins"] == 1          # tol 1.5 > gap 1.0 → merge


def test_cluster_empty():
    cl = cluster_basins(np.zeros((0, 6)), link_tol=0.5)
    assert cl["n_basins"] == 0 and cl["n_points"] == 0


# ───────────────────────────── held-out overlay (orphan / assigned) ─────────────────────────────
def test_assign_to_basins_orphan_vs_assigned():
    cents = [[0.0] * 6, [2.0, 0, 0, 0, 0, 0]]
    near = assign_to_basins(np.array([0.1, 0, 0, 0, 0, 0.0]), cents, link_tol=0.5)
    assert near["nearest_basin"] == 0 and near["orphan"] is False
    far = assign_to_basins(np.array([5.0, 0, 0, 0, 0, 0.0]), cents, link_tol=0.5)
    assert far["orphan"] is True                                    # farther than link_tol from every centroid ⇒ OOD


# ───────────────────────────── verdict (the discriminating gate) ─────────────────────────────
def _pooled(n_basins, inter, intra):
    return {"n_basins": n_basins, "min_inter_basin_dist": inter, "max_intra_nn_hop": intra,
            "centroids": [[0.0] * 6 for _ in range(max(1, n_basins))], "basin_sizes": [1] * max(1, n_basins)}


def test_verdict_multimodal_on_well_separated_basins():
    v = multimodality_verdict({"s1": {"centroid": {"delivers": True}}},
                              _pooled(2, inter=1.6, intra=0.4), {"s4": {"orphan": False}})
    assert v["verdict"] == "MULTIMODAL_BASINS_PRESENT" and v["justifies_k_head"] is True
    assert v["pooled_well_separated"] is True and v["blocker_if_not"] is None


def test_verdict_multimodal_when_centroid_fails_even_if_one_cluster():
    # one connected cluster, but the acceptable-set centroid does NOT deliver ⇒ averaging is harmful ⇒ multimodal fix
    v = multimodality_verdict({"s1": {"centroid": {"delivers": False}}},
                              _pooled(1, inter=None, intra=0.4), {"s4": {"orphan": False}})
    assert v["verdict"] == "MULTIMODAL_BASINS_PRESENT"
    assert v["states_with_nondelivering_centroid"] == ["s1"]


def test_verdict_single_cluster_is_representation_blocker():
    v = multimodality_verdict({"s1": {"centroid": {"delivers": True}}, "s3": {"centroid": {"delivers": True}}},
                              _pooled(1, inter=None, intra=0.5), {"s4": {"orphan": False}})
    assert v["verdict"] == "SINGLE_CONNECTED_CLUSTER"
    assert v["modality_is_blocker"] is False
    assert v["blocker_if_not"] == "REPRESENTATION_NOT_PROPOSAL_MODALITY_IS_BLOCKER"


def test_verdict_not_separated_when_inter_below_intra():
    # two nominal basins but the gap is NOT bigger than intra-basin spread ⇒ not genuinely separated
    v = multimodality_verdict({"s1": {"centroid": {"delivers": True}}},
                              _pooled(2, inter=0.3, intra=0.8), {"s4": {"orphan": False}})
    assert v["pooled_well_separated"] is False
    assert v["verdict"] == "SINGLE_CONNECTED_CLUSTER"


def test_verdict_flags_held_out_ood_orphan():
    v = multimodality_verdict({"s1": {"centroid": {"delivers": False}}},
                              _pooled(2, inter=1.6, intra=0.4),
                              {"s4": {"orphan": True}, "s7": {"orphan": False}})
    assert v["held_out_ood_warning"] is True
    assert v["held_out_orphans_outside_dev_basins"] == ["s4"]
