"""Reusable diagnostic-overlay video framework — render the EXACT rollout the evaluator scored, with legible overlays.

The rule (user, 2026-07-24): the video renders the SAME committed option / trajectory the scoring code executed — never a
separate "demo controller" or a hand-picked path. Overlays answer one question: *why did this rollout win or fail?* — a
top status bar, one or two load-bearing time-series (distance / dwell / K6-progress, NOT raw reward), a proposal/allocation
panel, and a certificate badge. Everything else goes in the per-clip manifest, not on the pixels.

Design: a small Strategy stack — `Panel` implementations draw onto a PIL frame; `TimeSeriesPanel` pre-renders its curve
ONCE (matplotlib) and only moves a cursor per frame (fast). MP4 via imageio (installed), GIF fallback via Pillow (no dep).
Reused by every clip in the video package (6D-1 critical pair / route montage / equal-budget, object variants, clamp-vs-
balltip) — do not re-implement per clip.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A truetype font (matplotlib bundles DejaVuSans) with a graceful fallback to PIL's bitmap default."""
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:  # noqa: BLE001 — font availability is environment-dependent; the default keeps rendering working
        return ImageFont.load_default()


_BADGE = {"SUCCESS": (34, 160, 80), "DELIVERED": (34, 160, 80), "FAIL": (196, 60, 60),
          "COLLISION": (210, 120, 20), "RUNNING": (70, 70, 90)}


@dataclass
class StatusBar:
    """Top status strip: `TASK | CTRL | B=.. | mode=.. | STATUS`. `status_fn(t)` returns the live status token."""

    text: str
    status_fn: Callable[[int], str]
    height: int = 34

    def draw(self, img: Image.Image, t: int) -> None:
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, self.height], fill=(20, 20, 28))
        f = _font(16)
        d.text((8, 8), self.text, fill=(235, 235, 240), font=f)
        st = self.status_fn(t)
        col = next((c for k, c in _BADGE.items() if k in st), (200, 200, 200))
        w = int(d.textlength(st, font=f))
        d.rectangle([img.width - w - 18, 5, img.width - 6, self.height - 5], fill=col)
        d.text((img.width - w - 12, 8), st, fill=(255, 255, 255), font=f)


@dataclass
class TimeSeriesPanel:
    """Bottom-right load-bearing curve(s) with a moving time cursor + optional phase/event vlines. NOT raw reward."""

    series: dict[str, np.ndarray]           # label -> per-frame values
    title: str = "distance to goal"
    vlines: Sequence[tuple[int, str]] = ()  # (frame, label) phase markers
    threshold: float | None = None          # e.g. the reach threshold, drawn as a dashed line
    size: tuple[int, int] = (300, 150)
    _bg: Image.Image | None = field(default=None, repr=False)

    def _render_bg(self) -> Image.Image:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(self.size[0] / 100, self.size[1] / 100), dpi=100)
        for lab, ys in self.series.items():
            ax.plot(np.arange(len(ys)), ys, lw=1.6, label=lab)
        for fr, lab in self.vlines:
            ax.axvline(fr, color="0.5", lw=0.8, ls=":")
            ax.text(fr, ax.get_ylim()[1], lab, fontsize=6, rotation=90, va="top", color="0.4")
        if self.threshold is not None:
            ax.axhline(self.threshold, color="#3a7", lw=1.0, ls="--")
        ax.set_title(self.title, fontsize=8)
        ax.tick_params(labelsize=6)
        if len(self.series) > 1:
            ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=False)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def draw(self, img: Image.Image, t: int) -> None:
        if self._bg is None:
            self._bg = self._render_bg()
        panel = self._bg.copy()
        n = max(1, max((len(v) for v in self.series.values()), default=1))
        x = int(8 + (panel.width - 16) * min(t, n - 1) / max(1, n - 1))   # cursor within the plot axes (approx)
        ImageDraw.Draw(panel).line([(x, 4), (x, panel.height - 4)], fill=(210, 40, 40), width=2)
        img.paste(panel, (img.width - panel.width - 6, img.height - panel.height - 6))


@dataclass
class InfoPanel:
    """A small text panel (bottom-left by default): proposal modes/probs, integer budget allocation, or a certificate
    checklist. `lines_fn(t)` may vary the content with time (e.g. K6 progress)."""

    lines_fn: Callable[[int], list[str]]
    corner: str = "bl"                       # bl | tl
    width: int = 250

    def draw(self, img: Image.Image, t: int) -> None:
        lines = self.lines_fn(t)
        f = _font(13)
        h = 6 + 17 * len(lines)
        d = ImageDraw.Draw(img, "RGBA")
        y0 = img.height - h - 6 if self.corner == "bl" else 40
        d.rectangle([6, y0, 6 + self.width, y0 + h], fill=(15, 15, 22, 190))
        for i, ln in enumerate(lines):
            d.text((12, y0 + 4 + 17 * i), ln, fill=(230, 230, 235), font=f)


def overlay_frames(frames: list[np.ndarray], panels: Sequence[Any]) -> list[Image.Image]:
    """Composite each panel (Strategy `draw(img, t)`) onto every frame. Returns PIL images."""
    out = []
    for t, fr in enumerate(frames):
        img = Image.fromarray(np.asarray(fr, np.uint8)).convert("RGB")
        for p in panels:
            p.draw(img, t)
        out.append(img)
    return out


def hstack(clip_a: list[Image.Image], clip_b: list[Image.Image], gap: int = 8) -> list[Image.Image]:
    """Side-by-side two overlaid clips (padded to equal length by freezing the shorter's last frame)."""
    n = max(len(clip_a), len(clip_b))
    h = max(clip_a[0].height, clip_b[0].height)

    def at(clip, i):
        return clip[min(i, len(clip) - 1)]
    out = []
    for i in range(n):
        a, b = at(clip_a, i), at(clip_b, i)
        canvas = Image.new("RGB", (a.width + gap + b.width, h), (0, 0, 0))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + gap, 0))
        out.append(canvas)
    return out


def summary_card(size: tuple[int, int], title: str, rows: list[tuple[str, str]], hold: int = 60) -> list[Image.Image]:
    """A held end-card (`hold` frames): a title + key/value rows — the 'professional' close the user asked for."""
    img = Image.new("RGB", size, (16, 16, 24))
    d = ImageDraw.Draw(img)
    d.text((24, 20), title, fill=(240, 240, 245), font=_font(22))
    for i, (k, v) in enumerate(rows):
        d.text((28, 66 + 26 * i), f"{k}", fill=(150, 150, 165), font=_font(15))
        d.text((260, 66 + 26 * i), f"{v}", fill=(235, 235, 240), font=_font(15))
    return [img] * hold


# ─────────────────────────── VIDEO_TRACE_CONSISTENCY_V1 (mandatory demo gate) ───────────────────────────
# A demonstration video must film the SAME rollout the scorer stepped. The 2026-07-24 coin bug (the renderer's frame_hook
# read a DIFFERENT, un-stepped state instance than structured_carry_rollout stepped) produced a frozen video whose summary
# metrics were still correct — credible-looking but wrong. This gate makes that class of mismatch a hard failure, not a
# manual visual catch. It is the mirror of the gate-contamination bug (shared mutable state) — here, two state instances
# where there should be one. Every recorder (coin, 6D-1, and future pick-place / AIBO) calls it before encoding.
def render_state_signature(frame) -> int:
    """A cheap per-frame content fingerprint (used to detect whether frames actually change across the rollout)."""
    return int(np.asarray(frame, np.int64).sum())


def assert_trace_render_consistency(frames, telemetry, *, eps: float = 1e-3, label: str = "") -> dict:
    """VIDEO_TRACE_CONSISTENCY_V1 — the rendered frames and the rollout telemetry must describe the SAME run.

    Raises AssertionError when:
      * frame count ≠ telemetry length (frames and trace are not the same sequence);
      * the rollout is DYNAMIC (telemetry span > eps) but the frames are STATIC (all identical) — the deepcopy-mismatch
        bug: the camera filmed a different, un-stepped state than the rollout stepped;
      * a dynamic rollout's first and last frames are bit-identical.
    Returns a diagnostics dict on success. `telemetry` is any per-step scalar that varies in a non-static rollout
    (dtz for the coin, distance-to-goal for reach)."""
    frames = list(frames)
    telem = np.asarray(telemetry, np.float64).ravel()
    n = len(frames)
    if n != len(telem):
        raise AssertionError(f"{label}: VIDEO_TRACE mismatch — {n} frames vs {len(telem)} telemetry steps (not the same rollout)")
    if n == 0:
        return {"n": 0, "telem_span": 0.0, "frames_vary": False}
    telem_span = float(np.max(telem) - np.min(telem))
    frames_vary = len({render_state_signature(f) for f in frames}) > 1
    if telem_span > eps and not frames_vary:
        raise AssertionError(f"{label}: VIDEO_TRACE mismatch — telemetry moves (span {telem_span:.4f}) but every frame is "
                             "IDENTICAL; the renderer filmed a different state than the rollout stepped (deepcopy mismatch).")
    if telem_span > eps and n >= 2 and np.array_equal(np.asarray(frames[0]), np.asarray(frames[-1])):
        raise AssertionError(f"{label}: VIDEO_TRACE mismatch — dynamic rollout but first==last frame bit-identical.")
    return {"n": n, "telem_span": round(telem_span, 4), "frames_vary": frames_vary}


def rollout_trace_hash(telemetry, final_metrics: dict) -> str:
    """A deterministic short hash of the rollout's per-step telemetry + final metrics — stored in the video manifest so
    the clip's rollout can be matched against the evaluator's record (`video_manifest.rollout_hash == evaluator.hash`)."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(telemetry, np.float64).ravel().tobytes())
    h.update(repr(sorted((k, round(float(v), 6) if isinstance(v, (int, float)) else v) for k, v in final_metrics.items())).encode())
    return h.hexdigest()[:16]


def encode_clip(frames: list[Image.Image], path: str | Path, fps: int = 30) -> str:
    """Encode to MP4 (imageio, installed) or GIF (Pillow, no dep) by suffix. Returns the path written."""
    path = Path(path)
    arrs = [np.asarray(f.convert("RGB"), np.uint8) for f in frames]
    if path.suffix == ".mp4":
        import imageio.v2 as imageio
        imageio.mimsave(str(path), arrs, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    else:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return str(path)
