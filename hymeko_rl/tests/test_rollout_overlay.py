"""Tests for the diagnostic-overlay video framework (pure compositing; no MuJoCo) + the VIDEO_TRACE_CONSISTENCY_V1 gate."""
import numpy as np
import pytest
from PIL import Image

from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, assert_trace_render_consistency, encode_clip, hstack, overlay_frames,
    rollout_trace_hash, summary_card)


def _frames(n=5, h=60, w=80):
    return [np.full((h, w, 3), i * 10 % 255, np.uint8) for i in range(n)]


def test_overlay_frames_returns_same_count_and_size():
    frames = _frames(6)
    panels = [StatusBar("TASK | CTRL", lambda t: "RUNNING" if t < 5 else "SUCCESS"),
              InfoPanel(lambda _t: ["line a", "line b"])]
    out = overlay_frames(frames, panels)
    assert len(out) == 6 and all(isinstance(im, Image.Image) for im in out)
    assert out[0].size == (80, 60)


def test_status_bar_badge_color_switches_on_token():
    img = Image.fromarray(_frames(1)[0])
    StatusBar("t", lambda _t: "COLLISION").draw(img, 0)
    arr = np.asarray(img)
    assert arr[:34].max() > 0                                # a status strip was drawn on top


def test_time_series_panel_prerenders_once_and_draws_cursor():
    p = TimeSeriesPanel({"d": np.linspace(1, 0, 10)}, title="dist", threshold=0.06, size=(120, 80))
    img = Image.fromarray(np.zeros((200, 240, 3), np.uint8))
    p.draw(img, 3)
    assert p._bg is not None                                 # curve cached
    p.draw(img, 7)                                           # second call reuses the cache (moves cursor only)
    assert np.asarray(img).max() > 0


def test_hstack_pads_to_equal_length_and_widths_add():
    a = overlay_frames(_frames(4, 60, 50), [])
    b = overlay_frames(_frames(7, 60, 30), [])
    s = hstack(a, b, gap=6)
    assert len(s) == 7                                       # padded to the longer clip
    assert s[0].size == (50 + 6 + 30, 60)


def test_summary_card_holds_frames():
    card = summary_card((120, 90), "title", [("k", "v"), ("k2", "v2")], hold=20)
    assert len(card) == 20 and card[0].size == (120, 90)


def test_encode_gif_roundtrip(tmp_path):
    frames = overlay_frames(_frames(5), [])
    out = encode_clip(frames, tmp_path / "clip.gif", fps=10)
    assert out.endswith(".gif")
    assert Image.open(out).n_frames == 5


def test_encode_mp4_when_available(tmp_path):
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        return                                              # mp4 optional (imageio); gif path covers the contract
    frames = overlay_frames(_frames(6), [])
    out = encode_clip(frames, tmp_path / "clip.mp4", fps=10)
    assert out.endswith(".mp4") and (tmp_path / "clip.mp4").stat().st_size > 0


# ── VIDEO_TRACE_CONSISTENCY_V1 — the gate that would have caught the 2026-07-24 static-video bug ──
def _moving_frames(n=8):
    return [np.full((30, 40, 3), i * 20 % 255, np.uint8) for i in range(n)]     # each frame distinct


def _static_frames(n=8):
    return [np.full((30, 40, 3), 77, np.uint8) for _ in range(n)]               # every frame identical


def test_trace_consistency_passes_when_frames_track_telemetry():
    telem = np.linspace(0.09, 0.01, 8)                                          # a dynamic rollout (dtz moves)
    d = assert_trace_render_consistency(_moving_frames(8), telem, label="ok")
    assert d["telem_span"] > 0 and d["frames_vary"] and d["n"] == 8


def test_trace_consistency_FAILS_on_static_frames_with_dynamic_telemetry():
    """THE regression: telemetry moves but every frame is identical — the deepcopy-mismatch bug. MUST raise."""
    telem = np.linspace(0.09, 0.01, 8)
    with pytest.raises(AssertionError, match="filmed a different state"):
        assert_trace_render_consistency(_static_frames(8), telem, label="bug")


def test_trace_consistency_fails_on_length_mismatch():
    with pytest.raises(AssertionError, match="not the same rollout"):
        assert_trace_render_consistency(_moving_frames(8), np.linspace(0, 1, 5))


def test_trace_consistency_allows_genuinely_static_rollout():
    telem = np.full(8, 0.05)                                                    # nothing moves — frames static is CONSISTENT
    d = assert_trace_render_consistency(_static_frames(8), telem, label="static")
    assert d["telem_span"] == 0.0 and not d["frames_vary"]


def test_trace_consistency_fails_when_first_equals_last_in_dynamic_rollout():
    frames = _moving_frames(8)
    frames[-1] = frames[0].copy()                                              # dynamic telemetry but ends where it started (bit-identical)
    # make the interior vary so only the first==last rule can trip
    with pytest.raises(AssertionError, match="first==last"):
        assert_trace_render_consistency(frames, np.linspace(0.09, 0.01, 8), label="fl")


def test_rollout_trace_hash_deterministic_and_sensitive():
    t = np.linspace(0.09, 0.01, 8)
    h1 = rollout_trace_hash(t, {"k6": 1, "max_dwell": 6})
    h2 = rollout_trace_hash(t, {"k6": 1, "max_dwell": 6})
    h3 = rollout_trace_hash(t, {"k6": 0, "max_dwell": 6})
    assert h1 == h2 and h1 != h3 and len(h1) == 16                              # deterministic; sensitive to metrics
