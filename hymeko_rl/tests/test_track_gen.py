"""Tests for Bézier track generation (`env/track_gen.py`) + its race-car integration.

Run: pytest -p no:randomly hymeko_rl/tests/test_track_gen.py"""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.locomotion_env import make_race_car
from hymeko_rl.env.track_gen import bezier_track, cubic_bezier, race_circuit


def test_cubic_bezier_hits_endpoints() -> None:
    p0, p1, p2, p3 = (np.array(v, dtype=float) for v in [(0, 0), (1, 2), (3, 2), (4, 0)])
    assert np.allclose(cubic_bezier(p0, p1, p2, p3, 0.0), p0)
    assert np.allclose(cubic_bezier(p0, p1, p2, p3, 1.0), p3)
    mid = cubic_bezier(p0, p1, p2, p3, 0.5)
    assert mid.shape == (2,) and np.isfinite(mid).all()


def test_bezier_track_passes_through_anchors() -> None:
    anchors = np.array([(0, 0), (4, 0), (4, 4), (0, 4)], dtype=float)
    spp = 20
    wp = bezier_track(anchors, samples_per_segment=spp, closed=True)
    assert wp.shape == (len(anchors) * spp, 2)
    # each segment starts (t=0) exactly on its anchor
    for i in range(len(anchors)):
        assert np.allclose(wp[i * spp], anchors[i])


def test_bezier_track_is_smooth_bounded_curvature() -> None:
    """The Bézier spline has no sharp reversals: consecutive heading changes stay small (unlike the polygon of
    anchors, whose 90° corners would jump ~π/2)."""
    anchors = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], dtype=float)   # a square
    wp = bezier_track(anchors, samples_per_segment=24, closed=True)
    d = np.diff(np.vstack([wp, wp[0]]), axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    assert float(np.abs(np.diff(ang)).max()) < 0.5      # smooth: no near-90°/180° per-step jumps


def test_race_circuit_scales_linearly() -> None:
    a, b = race_circuit(scale=10.0), race_circuit(scale=20.0)
    assert a.shape == b.shape
    assert np.allclose(2.0 * a, b)


def test_make_race_car_circuit_drivable_and_upright() -> None:
    env = make_race_car(course="circuit", circuit_scale=50.0, max_steps=4000)
    assert env.model.nhfield == 0                       # flat: no terrain
    assert len(env._track) > 100                        # a dense Bézier loop
    env.reset(seed=0)
    minup = 1.0
    for _ in range(4000):
        _, _, term, trunc, info = env.step(env.expert_action)
        minup = min(minup, info["upright"])
        assert np.isfinite(env.data.qacc).all()
        if term or trunc:
            break
    assert minup > 0.5, f"race car spun/flipped on the circuit (min upright {minup:.2f})"
    assert env._wp >= 5, "race car should make progress along the circuit"
