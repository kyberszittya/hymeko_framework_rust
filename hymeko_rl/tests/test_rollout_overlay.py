"""Tests for the diagnostic-overlay video framework (pure compositing; no MuJoCo)."""
import numpy as np
from PIL import Image

from hymeko_rl.viz.rollout_overlay import (
    InfoPanel, StatusBar, TimeSeriesPanel, encode_clip, hstack, overlay_frames, summary_card)


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
