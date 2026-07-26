"""Tests for the coin 6-D torque-θ option adapter (teacher-to-RL campaign).

Stage 0 layer here: frozen θ semantics, the θ normaliser (round-trip + always-legal), and the ANTI-ALIASING contract —
the Bellman action is the proposal centre θ_0, never the search-selected θ_exec. Later stages (dataset split isolation,
provenance under real physics) add to this file as they are built.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.semantics import (
    DELIVERY_CFG, DIM, ThetaBox, ThetaProvenance, option_semantics, theta_bounds)
from hymeko_rl.coin_delivery.theta_option.search import ThetaCandidateGenerator
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.option_rl.core import OptionReplayBuffer, OptionTransition
from hymeko_rl.option_rl.proposal import FixedBudgetSearch


# ───────────────────────────── θ box normaliser ─────────────────────────────
def test_theta_box_roundtrip_identity_on_legal_theta():
    box = ThetaBox()
    lo, hi = theta_bounds()
    rng = np.random.default_rng(0)
    thetas = lo + rng.random((200, DIM)) * (hi - lo)          # legal θ inside the box
    for t in thetas:
        z = box.norm(t)
        assert np.all(z >= -1.0 - 1e-6) and np.all(z <= 1.0 + 1e-6)
        back = box.denorm(z)
        assert np.allclose(back, t, atol=1e-5), (t, back)


def test_theta_box_denorm_is_always_legal_even_off_box():
    box = ThetaBox()
    lo, hi = theta_bounds()
    rng = np.random.default_rng(1)
    z = rng.uniform(-4.0, 4.0, (500, DIM))                    # z far outside [-1,1]
    theta = np.asarray([box.denorm(zi) for zi in z])
    assert np.all(theta >= lo - 1e-6) and np.all(theta <= hi + 1e-6)


def test_theta_box_clip_pulls_out_of_box_into_box():
    box = ThetaBox()
    lo, hi = theta_bounds()
    over = hi + 5.0
    under = lo - 5.0
    assert np.allclose(box.clip(over), hi, atol=1e-6)
    assert np.allclose(box.clip(under), lo, atol=1e-6)


# ───────────────────────────── frozen semantics ─────────────────────────────
def test_option_semantics_matches_frozen_delivery_config():
    sem = option_semantics()
    assert sem["dim"] == DIM == 6
    assert len(sem["components"]) == 6
    lo, hi = theta_bounds()
    for i, c in enumerate(sem["components"]):
        assert c["lo"] == float(lo[i]) and c["hi"] == float(hi[i])
    assert tuple(DELIVERY_CFG.lo) == (0.0, 0.0, -0.10, 1.0, 4.0, 0.0)
    assert tuple(DELIVERY_CFG.hi) == (0.25, 0.30, 0.10, 28.0, 48.0, 4.0)
    assert DELIVERY_CFG.horizon == 60 and DELIVERY_CFG.deliver is True
    k6 = sem["termination_and_k6"]
    assert k6["CENTER_TOL_m"] == CENTER_TOL == 0.02
    assert k6["SETTLE_VEL_mps"] == SETTLE_VEL == 0.06
    assert k6["HELD_DWELL_steps"] == HELD_DWELL == 6
    assert sem["frozen_split"]["development"] == ["s1", "s3"]
    assert sem["frozen_split"]["held_out"] == ["s4", "s7"]


# ───────────────────────────── ANTI-ALIASING (Bellman action = θ_0, not θ_exec) ─────────────────────────────
class _StubScorer:
    """Physics-free scorer for the structural aliasing test: prefers candidates NEAR a hidden target θ (so the search
    provably MOVES away from an off-target centre). Conforms to option_rl.CandidateScorer."""

    def __init__(self, target):
        self.target = np.asarray(target, np.float64)

    def score(self, candidate, rng):
        d = float(np.linalg.norm(np.asarray(candidate, np.float64) - self.target))
        return -d, {"k6_delivered": bool(d < 1e-3), "dtz_end": d}


def test_fixed_budget_search_preserves_center_and_moves_selected():
    """The search must (a) keep the input centre as `.center` unchanged and (b) be able to select a DIFFERENT θ_exec —
    if selected silently overwrote center, this test would fail."""
    box = ThetaBox()
    lo, hi = theta_bounds()
    center = box.clip(lo + 0.2 * (hi - lo))                   # an off-target centre
    target = box.clip(lo + 0.8 * (hi - lo))                   # where the search should pull toward
    gen = ThetaCandidateGenerator(box=box)
    sel = FixedBudgetSearch(generator=gen, scorer=_StubScorer(target), budget=32).select(center, np.random.default_rng(3))
    assert np.allclose(sel.center, center, atol=1e-6)         # centre (Bellman action) preserved
    assert not np.allclose(sel.selected, sel.center, atol=1e-4)   # θ_exec genuinely differs
    # θ_exec is closer to target than θ_0 was ⇒ the search is load-bearing
    assert np.linalg.norm(sel.selected - target) < np.linalg.norm(center - target)


def test_budget_zero_executes_center_directly():
    box = ThetaBox()
    center = box.clip(theta_bounds()[0] + 0.3)
    sel = FixedBudgetSearch(generator=ThetaCandidateGenerator(box=box), scorer=_StubScorer(center),
                            budget=0).select(center, np.random.default_rng(0))
    assert np.allclose(sel.selected, sel.center, atol=1e-6) and sel.budget == 0


def test_theta_provenance_keeps_center_as_bellman_action():
    box = ThetaBox()
    center = box.clip(theta_bounds()[0] + 0.1)
    selected = box.clip(theta_bounds()[0] + 0.25)
    prov = ThetaProvenance(center=center, selected=selected, score=1.0, budget=8,
                           outcome={"k6_delivered": True, "dtz_end": 0.01})
    d = prov.as_dict()
    assert d["theta_center"] == [round(float(x), 5) for x in center]
    assert d["theta_selected"] == [round(float(x), 5) for x in selected]
    assert prov.displacement(box) > 0.0                      # they differ


def test_replay_buffer_bellman_action_is_center_not_selected():
    """End-to-end anti-aliasing through the engine: the OptionTransition.action fed to the replay buffer is θ_0; θ_exec
    lives only in provenance. Sampling the batch returns θ_0 — never θ_exec. A regression that logged θ_exec as the
    action would flip this assertion."""
    box = ThetaBox()
    center = box.norm(box.clip(theta_bounds()[0] + 0.15))     # normalised θ_0 (the network action space)
    selected = box.norm(box.clip(theta_bounds()[0] + 0.30))   # a DIFFERENT θ_exec
    buf = OptionReplayBuffer()
    for _ in range(8):
        buf.add(OptionTransition(s=np.zeros(4, np.float32), action=center, reward=1.0, tau=60.0,
                                 s_next=np.zeros(4, np.float32), terminal=1.0, end="handoff",
                                 provenance={"theta_center": center, "theta_selected": selected}))
    bs, ba, br, bt, bs2, bd = buf.sample(4, np.random.default_rng(0))
    for a in ba.numpy():
        assert np.allclose(a, center, atol=1e-6)              # Bellman action == θ_0
        assert not np.allclose(a, selected, atol=1e-4)        # and NOT θ_exec
    # provenance retains θ_exec separately
    assert np.allclose(buf.provenance[0]["theta_selected"], selected, atol=1e-6)
