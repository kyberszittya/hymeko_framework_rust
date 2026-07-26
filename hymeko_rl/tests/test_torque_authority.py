"""Unit tests for the Route-B slew-admissible Δτ authority gate logic (pure parts).

The MuJoCo identification path (``nominal_dtau``, ``apply_dtau_step``, ``identify_Btau``, ``analyse_state_btau``) is
exercised at production scale by ``horizon_authority_benchmark.py --route-b`` (4 certified states, both ε, artifact +
figure) — the integration exercise. Here we pin the pure campaign-gate decision and the reproducibility metric.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.torque_authority import _repro_gap, decide_btau_route


def test_repro_gap_identical_and_different():
    Ba = np.zeros((4, 4))
    Ba[:, 0], Ba[:, 2] = [40, 0, 0, 0], [0, 30, 0, 0]
    assert _repro_gap(Ba, Ba.copy(), [True, False, True, False], [True, False, True, False]) == 0.0
    Bb = Ba * 2.0
    assert _repro_gap(Ba, Bb, [True, False, True, False], [True, False, True, False]) > 0.3
    assert _repro_gap(Ba, Bb, [False, False, False, False], [False, False, False, False]) is None  # no shared valid col


def test_decide_btau_route_established_needs_dev_and_heldout():
    def st(usable):
        return {"usable_authority": usable}
    # ≥1 dev AND ≥1 held-out usable → established
    res = {0: st(True), 1: st(False), 2: st(True), 3: st(False)}
    assert decide_btau_route(res, [0, 1], [2, 3])["route"] == "ROUTE_B_SLEW_ADMISSIBLE_DTAU_ESTABLISHED"
    # dev usable but NO held-out usable → dev-only
    res2 = {0: st(True), 1: st(True), 2: st(False), 3: st(False)}
    assert decide_btau_route(res2, [0, 1], [2, 3])["route"] == "ROUTE_B_DEV_ONLY"
    # none usable → not established
    res3 = {0: st(False), 1: st(False), 2: st(False), 3: st(False)}
    assert decide_btau_route(res3, [0, 1], [2, 3])["route"] == "AUTHORITY_RECOVERY_NOT_ESTABLISHED"


def test_decide_btau_route_all_four_usable():
    res = {i: {"usable_authority": True} for i in range(4)}
    out = decide_btau_route(res, [0, 1], [2, 3])
    assert out["route"] == "ROUTE_B_SLEW_ADMISSIBLE_DTAU_ESTABLISHED"
    assert out["dev_usable"] == [True, True] and out["heldout_usable"] == [True, True]
