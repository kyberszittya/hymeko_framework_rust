"""Tests for RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1 (label-only diagnostic)."""
import numpy as np
import pytest

from hymeko_rl.coin_delivery.coin_residual_hold_sweep import (
    frozen_isotropic_dirs,
    hold_candidates,
    paired_bootstrap_vs_k1,
    sweep_group,
    sweep_verdict,
)

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


# ── deterministic candidates ──
def test_frozen_isotropic_dirs_deterministic_and_unit():
    a, b = frozen_isotropic_dirs(3), frozen_isotropic_dirs(3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))          # frozen: identical across calls
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-5 for v in a)


def test_hold_candidates_deterministic_state_independent():
    c1, c2 = hold_candidates(3), hold_candidates(3)
    assert [n for n, _d, _m in c1] == [n for n, _d, _m in c2]
    assert all(np.array_equal(d1, d2) for (_n1, d1, _m1), (_n2, d2, _m2) in zip(c1, c2))
    assert c1[0][0] == "zero" and np.allclose(c1[0][1], 0.0)
    assert sum(np.allclose(d, 0.0) for _n, d, _m in c1) == 1        # single zero candidate
    assert len(c1) == 1 + 5 * (8 + 3)                              # 6 magnitudes, 8 signed bases + 3 isotropic
    for n, _d, _m in c1:
        assert "toward" not in n and "away" not in n               # no task-space direction labels


def test_hold_candidates_bounded_by_magnitude():
    for n, d, m in hold_candidates(3):
        assert np.max(np.abs(d)) <= 0.25 * m["magnitude"] + 1e-6


# ── verdict logic ──
def _paired(ci_by_k, lev_by_k):
    return {K: {"mean_paired_diff": (ci_by_k[K][0] + ci_by_k[K][1]) / 2, "ci95": list(ci_by_k[K]),
                "median_leverage": lev_by_k[K]} for K in ci_by_k}


def test_verdict_flat():
    p = _paired({1: (0, 0), 2: (-0.1, 0.1), 4: (-0.2, 0.15), 8: (-0.1, 0.2), 16: (-0.3, 0.1)},
                {1: 1.0, 2: 1.0, 4: 1.0, 8: 1.0, 16: 1.0})
    assert sweep_verdict(p) == "RESIDUAL_SIGNAL_FLAT_ACROSS_HOLD_HORIZON"


def test_verdict_increases():
    p = _paired({1: (0, 0), 2: (0.1, 0.3), 4: (0.4, 0.7), 8: (0.9, 1.4), 16: (1.5, 2.2)},
                {1: 1.0, 2: 1.3, 4: 1.7, 8: 2.2, 16: 2.9})
    assert sweep_verdict(p) == "RESIDUAL_SIGNAL_INCREASES_WITH_HOLD_HORIZON"


def test_verdict_finite_window():
    p = _paired({1: (0, 0), 2: (0.2, 0.5), 4: (0.6, 1.0), 8: (0.3, 0.8), 16: (-0.4, 0.4)},
                {1: 1.0, 2: 1.4, 4: 2.0, 8: 1.6, 16: 1.1})
    assert sweep_verdict(p) == "RESIDUAL_SIGNAL_HAS_FINITE_TEMPORAL_WINDOW"


def test_verdict_underpowered():
    p = _paired({1: (0, 0), 2: (-0.3, 0.35), 4: (-0.4, 0.42), 8: (-0.35, 0.4), 16: (-0.4, 0.45)},
                {1: 0.1, 2: 0.12, 4: 0.11, 8: 0.13, 16: 0.1})
    assert sweep_verdict(p) == "RESIDUAL_HOLD_SWEEP_UNDERPOWERED"


# ── env-touching (small) ──
@pytest.mark.slow
def test_hold_zero_residual_K_independent_and_deterministic():
    from hymeko_rl.coin_delivery.coin_counterfactual_labels import capture_state_panel
    from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    groups = capture_state_panel(pi0, range(6100, 6110), per_family=1, label=False)
    assert groups and all(g.gate_snap is not None for g in groups)
    rl = CoinRL4Dof()
    cands = hold_candidates(1)[:3]                                  # zero + two small
    per_k = sweep_group(rl, pi0, groups[0], cands, k_values=(1, 2, 4, 8))
    # zero residual (candidate 0) return is identical across K (a hold of the zero residual is pi_0 regardless of K)
    g0s = [per_k[K]["G"][0] for K in (1, 2, 4, 8)]
    assert max(abs(g - g0s[0]) for g in g0s) < 1e-9
    assert all(all(per_k[K]["det_ok"]) for K in per_k)              # x2 deterministic identity
    for K in per_k:
        for o in per_k[K]["outcomes"]:
            assert o["gate_active_steps"] <= K


@pytest.mark.slow
def test_paired_bootstrap_shapes():
    from hymeko_rl.coin_delivery.coin_counterfactual_labels import capture_state_panel
    from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    groups = capture_state_panel(pi0, range(6100, 6116), per_family=1, label=False)
    rl = CoinRL4Dof(); cands = hold_candidates(1)
    results = {g.group_id: sweep_group(rl, pi0, g, cands, k_values=(1, 2, 4)) for g in groups}
    paired = paired_bootstrap_vs_k1(groups, results, (1, 2, 4), n_boot=200)
    assert paired[1]["mean_paired_diff"] == 0.0                     # K=1 vs itself
    for K in (2, 4):
        assert len(paired[K]["ci95"]) == 2 and paired[K]["ci95"][0] <= paired[K]["ci95"][1]
