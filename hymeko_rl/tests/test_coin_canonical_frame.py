"""R1 canonical target-frame representation — the SIX mandatory equivariance tests (the user's correctness gate). The
canonicalisation must be carried through BOTH state and θ label: input invariance alone would leave the same canonical
input carrying two different labels (the aliasing the audit exposed)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.canonical_frame import (
    BALANCE_IDX, R1_GROUP_ORDER, canonicalise, flatten_r1, from_canonical_theta, perp_key, r1_feature_dim,
    swap_grouped, swap_theta, to_canonical_theta)
from hymeko_rl.coin_delivery.theta_option.semantics import DIM


def _grouped(seed=0):
    rng = np.random.default_rng(seed)
    g = {}
    for name in R1_GROUP_ORDER:
        # SIDE_* groups are length-2, SHARED_* length-1 (per r1_feature_dim accounting)
        from hymeko_rl.coin_delivery.theta_option.canonical_frame import R1_KIND, SHARED_ALONG, SHARED_PERP
        n = 1 if R1_KIND[name] in (SHARED_ALONG, SHARED_PERP) else 2
        g[name] = rng.normal(size=n)
    return g


# ── 1. swap twice == identity (state AND θ) ──
def test_swap_twice_is_identity_state_and_theta():
    g = _grouped(1)
    gg = swap_grouped(swap_grouped(g))
    for name in g:
        assert np.allclose(gg[name], g[name]), name
    theta = np.array([0.1, 0.2, 0.05, 12.0, 20.0, 1.0])
    assert np.allclose(swap_theta(swap_theta(theta)), theta)


# ── 2. canonical_features(x) == canonical_features(swap(x)) ──
def test_canonical_features_invariant_under_mirror():
    for seed in range(20):
        g = _grouped(seed)
        ca, _ = canonicalise(g)
        cb, _ = canonicalise(swap_grouped(g))
        assert np.allclose(flatten_r1(ca), flatten_r1(cb), atol=1e-9), seed


def test_perp_key_is_antisymmetric():
    g = _grouped(3)
    assert abs(perp_key(swap_grouped(g)) + perp_key(g)) < 1e-9   # key(Sg) == -key(g)


# ── 3. canonical θ transforms consistently ──
def test_canonical_theta_negates_balance_only_when_swapped():
    theta = np.array([0.1, 0.2, 0.07, 12.0, 20.0, 1.0])
    assert np.allclose(to_canonical_theta(theta, False), theta)                 # not swapped ⇒ unchanged
    cw = to_canonical_theta(theta, True)
    assert cw[BALANCE_IDX] == -theta[BALANCE_IDX]                               # balance negated
    assert np.allclose(np.delete(cw, BALANCE_IDX), np.delete(theta, BALANCE_IDX))  # others unchanged


# ── 4. decode(canonical prediction) recovers the physical θ (round-trip) ──
def test_theta_canonical_round_trip():
    rng = np.random.default_rng(4)
    for _ in range(10):
        theta = rng.uniform([-0.5] * DIM, [0.5] * DIM)
        for swapped in (False, True):
            assert np.allclose(from_canonical_theta(to_canonical_theta(theta, swapped), swapped), theta)


# ── 5. contact/authority (SIDE_PERP) features transform consistently ──
def test_side_perp_group_swaps_and_negates():
    g = _grouped(5)
    s = swap_grouped(g)
    # tip_coin_perp is SIDE_PERP: [L,R] -> [-R,-L]
    assert np.allclose(s["tip_coin_perp"], -g["tip_coin_perp"][::-1])
    # normal_along is SIDE_ALONG: [L,R] -> [R,L] (no sign change)
    assert np.allclose(s["normal_along"], g["normal_along"][::-1])
    # coin_vel_perp is SHARED_PERP: negated
    assert np.allclose(s["coin_vel_perp"], -g["coin_vel_perp"])
    # dtz is SHARED_ALONG: unchanged
    assert np.allclose(s["dtz"], g["dtz"])


# ── 6. streaming == batch canonicalisation (determinism) ──
def test_canonicalisation_is_deterministic_streaming_equals_batch():
    gs = [_grouped(s) for s in range(8)]
    single = [flatten_r1(canonicalise(g)[0]) for g in gs]
    # "batch" = recompute independently; must be identical (no shared/mutable state)
    again = [flatten_r1(canonicalise(g)[0]) for g in reversed(gs)][::-1]
    for a, b in zip(single, again):
        assert np.array_equal(a, b)
    assert r1_feature_dim() == len(single[0])                   # declared dim matches the flattened length
