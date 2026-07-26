"""R2 graph adapter — training-free contract tests: INFORMATION PARITY (every R1 v3 group recoverable, nothing added or
lost), MIRROR EQUIVALENCE (G(Mx)≅G(x) after canonicalisation), and deterministic canonical node order. The encoder-level
permutation invariance + θ-output equivariance are tested with the encoder (step 2)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.canonical_frame import (
    R1_GROUP_ORDER, canonicalise, group_len, swap_grouped)
from hymeko_rl.coin_delivery.theta_option.relational_graph import (
    build_graph_from_canonical, recover_r1_groups)


def _grouped(seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=group_len(name)) for name in R1_GROUP_ORDER}


# ── information parity: recover(build(g)) == g for EVERY R1 v3 group ──
def test_information_parity_every_group_recovered():
    g = _grouped(0)
    rec = recover_r1_groups(build_graph_from_canonical(g, was_swapped=False))
    missing = [n for n in R1_GROUP_ORDER if n not in rec]
    assert not missing, f"lost groups (no home in the graph): {missing}"
    for name in R1_GROUP_ORDER:
        assert np.allclose(np.asarray(rec[name]).ravel(), np.asarray(g[name]).ravel(), atol=1e-9), name


def test_no_new_information_beyond_r1_groups():
    # every recovered key is a real R1 group (nothing invented)
    rec = recover_r1_groups(build_graph_from_canonical(_grouped(1), was_swapped=True))
    assert set(rec).issubset(set(R1_GROUP_ORDER))


# ── mirror equivalence: G built from canonicalise(g) == G from canonicalise(swap(g)) ──
def test_graph_mirror_equivalence():
    for seed in range(15):
        g = _grouped(seed)
        ca, sa = canonicalise(g)
        cb, sb = canonicalise(swap_grouped(g))
        Ga = build_graph_from_canonical(ca, sa)
        Gb = build_graph_from_canonical(cb, sb)
        for t in Ga.nodes:
            assert np.allclose(Ga.nodes[t], Gb.nodes[t], atol=1e-9), (seed, t)   # identical canonical graph ATTRIBUTES
        assert np.allclose(Ga.goal, Gb.goal) and np.allclose(Ga.authority, Gb.authority) and np.allclose(Ga.bimanual, Gb.bimanual)
        # the decode flag CORRECTLY flips: a config and its mirror reach the same canonical graph but need opposite T_θ
        assert Ga.was_swapped != Gb.was_swapped


def test_per_side_nodes_are_two_rows_canonical_order():
    G = build_graph_from_canonical(_grouped(2), was_swapped=False)
    assert G.nodes["TIP"].shape[0] == 2 and G.nodes["CONTACT"].shape[0] == 2 and G.nodes["PORT"].shape[0] == 2
    assert G.nodes["COIN"].shape[0] == 1 and G.nodes["TARGET"].shape[0] == 1
    assert G.authority.shape[0] == 2                            # per-side authority edges


def test_build_is_deterministic():
    g = _grouped(3)
    a = build_graph_from_canonical(g, was_swapped=True)
    b = build_graph_from_canonical(g, was_swapped=True)
    assert np.array_equal(a.nodes["CONTACT"], b.nodes["CONTACT"]) and np.array_equal(a.bimanual, b.bimanual)
