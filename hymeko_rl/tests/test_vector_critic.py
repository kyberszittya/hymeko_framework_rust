"""Unit tests for the vector-critic branch pure pieces: SearchObjective component signals, cosine, and the
constraint-projected gradient (PCGrad-style)."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.train.search_objective import CONSTRAINTS, OBJECTIVES, SearchObjective
from hymeko_rl.train.vector_critic import cosine, projected_gradient


def test_search_objective_signals():
    so = SearchObjective(near_coin=0.06, progress_eps=0.002)
    # in contact, coin moved 0.02 toward zone under fingertip contact
    s = so.step_signals(prev_dist=0.30, dist=0.28, min_tip=0.02, both_contact=True,
                        fingertip_contact=True, arm_body_contact=False, in_zone=False)
    assert s["contact"] == 1.0 and s["progress"] == pytest.approx(0.02) and s["body_progress"] == 0.0
    assert s["antiexploit"] == pytest.approx(0.02) and s["approach"] > 0.6
    # body-only push: arm-body contact, no fingertip → progress attributed to body
    b = so.step_signals(prev_dist=0.30, dist=0.28, min_tip=0.5, both_contact=False,
                        fingertip_contact=False, arm_body_contact=True, in_zone=False)
    assert b["contact"] == 0.0 and b["progress"] == 0.0 and b["body_progress"] == pytest.approx(0.02)
    assert b["antiexploit"] == pytest.approx(-0.02) and b["approach"] == 0.0
    # in-zone delivery
    d = so.step_signals(prev_dist=0.05, dist=0.03, min_tip=0.02, both_contact=True,
                        fingertip_contact=True, arm_body_contact=False, in_zone=True)
    assert d["delivery"] == 1.0


def test_objective_constraint_split():
    assert set(OBJECTIVES) <= {"delivery", "progress"}
    assert CONSTRAINTS["contact"] == "up" and CONSTRAINTS["antiexploit"] == "up"
    assert CONSTRAINTS["body_progress"] == "down"


def test_cosine():
    assert cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(-1.0)
    assert cosine(np.array([1.0, 0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.zeros(2), np.array([1.0, 0])) == 0.0


def test_projected_gradient_respects_contact_constraint():
    # objective direction conflicts with contact (points opposite): projection must remove the conflict
    grads = {
        "delivery": np.array([1.0, 0.0]),
        "progress": np.array([0.0, 0.0]),
        "contact": np.array([-1.0, 0.0]),   # contact gradient opposes the objective → conflict
        "antiexploit": np.array([0.0, 1.0]),
        "body_progress": np.array([0.0, 0.0]),
    }
    g, info = projected_gradient(grads)
    assert info["projections_fired"].get("contact") is True
    assert np.dot(g, grads["contact"]) >= -1e-6      # no longer decreases contact


def test_projected_gradient_removes_body_progress_increase():
    grads = {
        "delivery": np.array([1.0, 1.0]),
        "progress": np.array([0.0, 0.0]),
        "contact": np.array([1.0, 0.0]),
        "antiexploit": np.array([1.0, 0.0]),
        "body_progress": np.array([0.0, 1.0]),   # objective aligns with +body_progress → must be projected out
    }
    g, info = projected_gradient(grads)
    assert info["projections_fired"].get("body_progress") is True
    assert np.dot(g, grads["body_progress"]) <= 1e-6      # no longer increases body-progress


def test_projected_gradient_no_conflict_passes_through():
    grads = {
        "delivery": np.array([1.0, 0.0]),
        "progress": np.array([1.0, 0.0]),
        "contact": np.array([1.0, 0.0]),      # aligned → no projection
        "antiexploit": np.array([1.0, 0.0]),
        "body_progress": np.array([0.0, -1.0]),  # objective already reduces body-progress
    }
    g, info = projected_gradient(grads)
    assert info["projections_fired"] == {}
    assert cosine(g, np.array([1.0, 0.0])) == pytest.approx(1.0)
