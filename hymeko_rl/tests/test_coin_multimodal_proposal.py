"""M1 K-head acceptable-set proposal — permutation-invariant set loss, legal modes, fair budget split (fast) + a slow
physical deploy. The loss must be order-invariant (a SET loss), must reward covering every mode and penalise a head that
sits between modes, and must only push heads apart when the target set is genuinely multimodal. The deploy must split a
fixed total budget K×(8/K) and keep centre-inclusion per mode."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import (
    KHeadProposal, KHeadProposalNet, KHeadTrainState, fit_khead, is_multimodal_target_set, set_loss)
from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox
from hymeko_rl.option_rl.proposal import allocate_budget


def _t(x):
    return torch.as_tensor(np.asarray(x, np.float32))


# ───────────────────────────── set loss (permutation-invariant) ─────────────────────────────
def test_set_loss_is_permutation_invariant_in_heads_and_targets():
    H = _t([[0.5, 0, 0, 0, 0, 0], [-0.5, 0, 0, 0, 0, 0]])
    T = _t([[0.5, 0, 0, 0, 0, 0], [-0.5, 0, 0, 0, 0, 0], [0.0, 0.2, 0, 0, 0, 0]])
    base = float(set_loss(H, T))
    assert abs(float(set_loss(H[[1, 0]], T)) - base) < 1e-6            # reorder heads
    assert abs(float(set_loss(H, T[[2, 0, 1]])) - base) < 1e-6         # reorder targets


def test_set_loss_penalises_a_head_between_two_modes():
    T = _t([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]])              # two modes at ±1
    on_modes = set_loss(_t([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]]), T)   # heads on the modes
    averaged = set_loss(_t([[0.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]]), T)    # heads at the (bad) average
    assert float(on_modes) < float(averaged)                          # the between-mode average is worse


def test_set_loss_diversity_penalty_only_on_multimodal_state():
    T_multi = _t([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]])
    collapsed = _t([[0.9, 0, 0, 0, 0, 0], [0.9, 0, 0, 0, 0, 0]])       # two heads collapsed together
    l_multi = float(set_loss(collapsed, T_multi, multimodal=True, diversity_weight=1.0))
    l_uni = float(set_loss(collapsed, T_multi, multimodal=False, diversity_weight=1.0))
    assert l_multi > l_uni                                             # collapse penalised only when multimodal


def test_is_multimodal_target_set():
    assert is_multimodal_target_set(np.array([[1.0] + [0] * 5, [-1.0] + [0] * 5])) is True     # spread > 0.8
    assert is_multimodal_target_set(np.array([[0.1] + [0] * 5, [0.15] + [0] * 5])) is False    # tight
    assert is_multimodal_target_set(np.array([[0.0] * 6])) is False                            # single point


# ───────────────────────────── K-head proposal (legal modes, uniform prob) ─────────────────────────────
def test_khead_modes_are_legal_and_uniform_prob():
    box = ThetaBox()
    prop = KHeadProposal(k=4, net=KHeadProposalNet(4), box=box)
    modes = prop.modes(np.zeros(42, np.float32))
    assert len(modes) == 4
    assert all(abs(m.prob - 0.25) < 1e-9 for m in modes)              # uniform
    assert [m.mode_id for m in modes] == [0, 1, 2, 3]
    for m in modes:                                                   # every centre is a legal (in-box) θ
        assert np.all(np.asarray(m.center) >= box.lo - 1e-5) and np.all(np.asarray(m.center) <= box.hi + 1e-5)


# ───────────────────────────── fair fixed-budget split (the user's K×(8/K)) ─────────────────────────────
def test_allocate_budget_is_the_even_split_for_uniform_prob():
    assert allocate_budget([1.0], 8) == [8]                           # B0: K=1 × 8
    assert allocate_budget([0.5, 0.5], 8) == [4, 4]                   # M2: K=2 × 4
    assert allocate_budget([0.25] * 4, 8) == [2, 2, 2, 2]            # M4: K=4 × 2
    assert sum(allocate_budget([1 / 3] * 3, 8)) == 8                  # total is always the budget


# ───────────────────────────── training covers both modes ─────────────────────────────
def test_fit_khead_k2_covers_a_two_mode_target_set():
    # one state whose acceptable set is two separated modes; K=2 must place one head near each (low recall)
    targets = np.array([[0.8, 0.2, 0, 0, 0, 0]] * 4 + [[-0.8, -0.2, 0, 0, 0, 0]] * 4, np.float32)
    st = KHeadTrainState("x", features=np.zeros(42, np.float32), targets_norm=targets,
                         multimodal=is_multimodal_target_set(targets))
    prop, info = fit_khead([st], k=2, epochs=800, lr=5e-3, seed=0)
    centers_norm = np.asarray([prop.box.norm(c) for c in prop._heads(np.zeros(42, np.float32))])
    # each target mode has a head within a small radius (both modes covered, not a single averaged head)
    for mode in (np.array([0.8, 0.2, 0, 0, 0, 0]), np.array([-0.8, -0.2, 0, 0, 0, 0])):
        assert min(np.linalg.norm(centers_norm - mode, axis=1)) < 0.3
    assert info["per_state_recall"]["x"] < 0.3


# ───────────────────────────── physical deploy (slow) ─────────────────────────────
@pytest.mark.slow
def test_multimodal_search_select_splits_budget_and_delivers_on_s1():
    """A 2-head proposal seeded with s1's canonical θ in BOTH heads deploys at total budget 8 → per-mode [4,4], and the
    centre-inclusive per-mode search delivers frozen K6 on s1 (a delivering dev cradle). Budget accounting is exact."""
    from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import multimodal_search_select
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    import json
    bank = json.load(open("reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"))
    canon = [e for e in bank["states"] if e["tag"] == "s1"][0]["canonical_theta_vec"]
    snap, _ = acquire_snapshot(load_harness(), 14250)
    box = ThetaBox()
    net = KHeadProposalNet(2)
    # hand-set both heads to s1's canonical (normalised) so modes() returns the delivering θ twice
    with torch.no_grad():
        z = torch.as_tensor(np.tile(box.norm(np.asarray(canon)), 2), dtype=torch.float32)
        net.heads.bias.copy_(torch.atanh(torch.clamp(z, -0.999, 0.999)))
        net.heads.weight.zero_()
    prop = KHeadProposal(k=2, net=net, box=box)
    dep = multimodal_search_select(snap, prop, np.zeros(42, np.float32), np.random.default_rng(0), budget=8)
    assert sum(dep.per_mode_budget) == 8 and dep.per_mode_budget == [4, 4]
    assert dep.provenance.outcome.get("delivery_success") is True
