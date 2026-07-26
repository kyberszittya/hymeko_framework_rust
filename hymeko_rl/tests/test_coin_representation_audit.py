"""Decision-time representation audit — pure diagnostics (no physics). These must correctly detect: whether feature
proximity predicts θ proximity (smoothness), the nearest-feature neighbour (for training-free retrieval), and the
left-right ordering deficit (the missing canonical frame)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.representation_audit import (
    audit_verdict, lipschitz_analysis, nearest_neighbour_by_feature, ordering_deficit, swap_lr)


def _flat(g):
    return np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])


# ───────────────────────────── smoothness ─────────────────────────────
def test_lipschitz_high_correlation_when_features_track_theta():
    # φ and θ move together ⇒ feature-proximity predicts θ-proximity ⇒ high correlation
    feats = {f"s{i}": np.array([float(i), 0.0]) for i in range(5)}
    thetas = {f"s{i}": np.array([float(i) * 0.1, 0.0]) for i in range(5)}
    r = lipschitz_analysis(feats, thetas)
    assert r["corr_dphi_dtheta"] > 0.9                          # feature-proximity predicts θ-proximity
    assert r["lipschitz_ratio"]["max"] < 1e6                    # finite, well-defined


def test_lipschitz_low_correlation_when_decoupled():
    rng = np.random.default_rng(0)
    feats = {f"s{i}": rng.normal(size=4) for i in range(6)}
    thetas = {f"s{i}": rng.normal(size=6) for i in range(6)}   # independent of features
    r = lipschitz_analysis(feats, thetas)
    assert abs(r["corr_dphi_dtheta"]) < 0.6                     # decoupled ⇒ weak correlation


# ───────────────────────────── nearest neighbour ─────────────────────────────
def test_nearest_neighbour_never_self_and_picks_closest():
    feats = {"a": np.array([0.0, 0]), "b": np.array([0.1, 0]), "c": np.array([5.0, 0])}
    nn = nearest_neighbour_by_feature(feats, ["a", "b", "c"], ["a", "b", "c"])
    assert nn["a"]["nn_tag"] == "b" and nn["b"]["nn_tag"] == "a"
    assert nn["c"]["nn_tag"] == "b"                             # closest of {a,b}
    assert all(nn[t]["nn_tag"] != t for t in feats)            # never self
    # held-out-style query: candidates restricted to a subset not containing the query
    held = nearest_neighbour_by_feature({**feats, "h": np.array([0.2, 0])}, ["h"], ["a", "b", "c"])
    assert held["h"]["nn_tag"] == "b"                           # h(0.2) nearest to b(0.1) among dev candidates


# ───────────────────────────── L/R ordering deficit ─────────────────────────────
def _grouped(fn, normal, xc_rel):
    # minimal grouped dict with the per-side contact groups + one shared group
    return {"dtz": np.array([0.5]), "fn": np.asarray(fn, float),
            "normal": np.asarray(normal, float), "xc_rel": np.asarray(xc_rel, float)}


def test_swap_lr_swaps_contact_pairs():
    g = _grouped(fn=[1.0, 2.0], normal=[1, 1, 2, 2], xc_rel=[3, 3, 4, 4])
    s = swap_lr(g, include_joints=False)
    assert list(s["fn"]) == [2.0, 1.0]                         # L↔R
    assert list(s["normal"]) == [2, 2, 1, 1]
    assert list(s["xc_rel"]) == [4, 4, 3, 3]
    assert list(s["dtz"]) == [0.5]                             # shared unchanged


def test_ordering_deficit_zero_for_symmetric_nonzero_for_asymmetric():
    sym = _grouped(fn=[1.0, 1.0], normal=[2, 2, 2, 2], xc_rel=[3, 3, 3, 3])       # L==R ⇒ swap is a no-op
    asym = _grouped(fn=[1.0, 9.0], normal=[1, 1, 2, 2], xc_rel=[3, 3, 4, 4])      # L≠R ⇒ deficit > 0
    d = ordering_deficit({"sym": sym, "asym": asym}, _flat)
    assert d["per_cradle"]["sym"]["contact_swap_deficit"] == 0.0
    assert d["per_cradle"]["asym"]["contact_swap_deficit"] > 0.0
    assert d["features_are_canonically_ordered"] is False       # a nonzero deficit exists ⇒ not canonical


# ───────────────────────────── verdict ─────────────────────────────
def test_audit_verdict_retrievable_dev_and_defects():
    lip = {"corr_dphi_dtheta": 0.8, "lipschitz_ratio": {"max": 2.0}}
    order = {"features_are_canonically_ordered": False}
    v = audit_verdict(6, 6, 2, 2, lip, order)                   # retrieval works on all dev folds
    assert v["retrieval_works_on_dev"] is True
    assert v["audit_summary"] == "CURRENT_42D_SUPPORTS_RETRIEVAL"
    assert "NO_CANONICAL_LEFT_RIGHT_ORDERING" in v["identified_defects"]


def test_audit_verdict_flags_non_learnable_map():
    lip = {"corr_dphi_dtheta": 0.05, "lipschitz_ratio": {"max": 20.0}}
    order = {"features_are_canonically_ordered": False}
    v = audit_verdict(0, 6, 0, 2, lip, order)                   # retrieval fails on dev
    assert v["retrieval_works_on_dev"] is False
    assert v["audit_summary"] == "CURRENT_42D_DOES_NOT_ADMIT_A_LEARNABLE_MAP_AS_IS"
    assert "FEATURE_PROXIMITY_DOES_NOT_PREDICT_THETA_TRANSFER_ON_DEV" in v["identified_defects"]
    assert "NON_SMOOTH_OR_COORDINATE_DEPENDENT_MAP" in v["identified_defects"]
