"""Tests for the Stage-3 controlled-forward-displacement primitive: pure predicate/score logic + one integration run.

The CEM search + Δτ rollout are exercised at production scale by ``horizon_authority_benchmark.py --forward`` (4 states,
artifact + figure + GIF). Here we pin the external success certificate and the CEM score's controlled-push shaping, plus
one integration run on development state s3 confirming a real controlled forward push.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.forward_displacement import ForwardConfig, _feasible, score, success


def _metrics(**kw):
    """A controlled-success metrics dict; override fields to probe each predicate clause."""
    m = {"forward": 0.05, "cross": 0.01, "total_disp": 0.051, "forward_at_release": 0.03, "peak_qdot": 1.2,
         "peak_coin_speed": 0.8, "terminal_coin_speed": 0.5, "contact_lost_steps": 2, "lost_before_release": 0,
         "release_step": 8, "straddle_min": -0.9, "min_fn_push": 1.5, "coin_trace": []}
    m.update(kw)
    return m


def test_feasible_excludes_motion_breach_and_backward():
    cfg = ForwardConfig()
    assert _feasible(_metrics(), cfg) is True
    assert _feasible(_metrics(peak_qdot=3.5), cfg) is False           # joint-velocity hard breach
    assert _feasible(_metrics(peak_coin_speed=1.6), cfg) is False     # coin-speed blow-up
    assert _feasible(_metrics(forward=-0.01), cfg) is False           # net backward


def test_success_requires_all_controlled_clauses():
    cfg = ForwardConfig()
    thr, passive = 0.02, 0.002       # forward 0.05 ≥ 0.02 and > 5×0.002=0.01
    assert success(_metrics(), thr, passive, cfg) is True
    assert success(_metrics(forward=0.015), thr, passive, cfg) is False          # below threshold
    assert success(_metrics(forward=0.008), 0.001, 0.002, cfg) is False          # not > 5× passive
    assert success(_metrics(cross=0.10), thr, passive, cfg) is False             # cross > forward
    assert success(_metrics(lost_before_release=1), thr, passive, cfg) is False  # grip lost during push
    assert success(_metrics(min_fn_push=0.01), thr, passive, cfg) is False       # push force below floor
    assert success(_metrics(terminal_coin_speed=1.3), thr, passive, cfg) is False  # throw (uncontrolled release)


def test_score_infeasible_is_neg_inf_and_penalises_flick():
    cfg = ForwardConfig()
    assert score(_metrics(peak_qdot=4.0), cfg) == -np.inf
    controlled = score(_metrics(lost_before_release=0, terminal_coin_speed=0.5), cfg)
    flick = score(_metrics(lost_before_release=5, terminal_coin_speed=1.4), cfg)
    assert controlled > flick                                         # a controlled push scores above a flick


@pytest.mark.slow
def test_cem_finds_controlled_push_s3():
    """Integration: on development state s3, the CEM finds a CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT."""
    from hymeko_rl.coin_delivery.contact_velocity import BvConfig, CradleSnapshot
    from hymeko_rl.coin_delivery.forward_displacement import cem_search
    from hymeko_rl.experiments.bv_identification_benchmark import _load_frozen, acquire_certified_straddle
    from hymeko_rl.experiments.video_coin_variants import _setup
    stack, cfg_coop, v2, mu = _load_frozen()
    pi0, base, forbidden = _setup()
    rl, saved, handoff, meta = acquire_certified_straddle(pi0, base, forbidden, v2, stack, cfg_coop, 14750, tries=3)
    assert rl is not None and meta["certified"]
    snap = CradleSnapshot(rl, stack, saved, handoff["prev_tau"], handoff["q_target"],
                          release_coin=True, coast_mu=mu, cfg=BvConfig())
    res = cem_search(snap, ForwardConfig())
    assert res["success"], f"expected a controlled forward push on s3, got {res['best_metrics']}"
    bm = res["best_metrics"]
    assert bm["forward"] > res["threshold"] and bm["lost_before_release"] == 0
