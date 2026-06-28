"""Episode evaluation harness — a goal / death / timeout scoreboard with plots and a rendered GIF.

Env-agnostic: an episode's outcome is read from the terminal ``info``/flags — ``death`` when
``terminated`` and ``info["death"]`` is set (a safety death / disk knocked out), ``goal`` when
``terminated`` otherwise (reached / disk-in-zone), else ``timeout`` (truncated). Points = the
episode's summed reward. Works for both :class:`ArmReachEnv` (safety task) and
:class:`PlanarGraspEnv` (Galambos), which both expose the gym 5-tuple + ``info["death"]``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ActionFn = Callable[[Any, np.ndarray], np.ndarray]


def now_stamp() -> str:
    """Local wall-clock label for stamping generated artifacts (GIFs, result tables).

    # Postconditions returns a ``"YYYY-MM-DD HH:MM:SS"`` string of the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def experiment_dir(base: str | Path, name: str) -> Path:
    """Create and return one **timestamped run directory** ``<base>/YYYY_MM_DD_HH_MM_<name>`` — the single
    home for an experiment run's artifacts (GIFs, plots, JSON, CSV). Every experiment writes *all* of its
    output here, so a run is self-contained and runs never clobber each other. The timestamp leads the name,
    so a directory listing sorts runs chronologically.

    # Postconditions the directory exists; its name begins with the minute-resolution local timestamp."""
    out = Path(base) / f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def results_to_csv(path: str | Path, results: "dict[str, dict[str, Any]]", *, key_col: str = "config") -> Path:
    """Write a ``{name: {metric: value, …}}`` result table as a CSV (one row per top-level key).

    The column order is ``key_col`` then the union of metric keys (in first-seen order). # Postconditions
    returns the written ``.csv`` path; an empty ``results`` writes just the header ``key_col``."""
    import csv

    columns: list[str] = []
    for metrics in results.values():
        for k in metrics:
            if k not in columns:
                columns.append(k)
    out = Path(path).with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([key_col, *columns])
        for name, metrics in results.items():
            writer.writerow([name, *(metrics.get(c, "") for c in columns)])
    return out


def _stamp_frames(frames: list[np.ndarray], label: str) -> list[np.ndarray]:
    """Draw ``label`` bottom-right (small dark box) on a copy of each RGB frame, for provenance.

    # Preconditions all frames are ``(H, W, 3)`` uint8. # Postconditions a new list, same shapes."""
    from PIL import Image, ImageDraw, ImageFont
    if not label or not frames:
        return frames
    try:
        font = ImageFont.load_default(size=12)     # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    out: list[np.ndarray] = []
    for f in frames:
        img = Image.fromarray(f).convert("RGB")
        draw = ImageDraw.Draw(img)
        w = int(draw.textlength(label, font=font))
        h_img = img.height
        pad = 3
        draw.rectangle((img.width - w - 2 * pad, h_img - 16 - pad, img.width, h_img), fill=(0, 0, 0))
        draw.text((img.width - w - pad, h_img - 15), label, fill=(235, 235, 235), font=font)
        out.append(np.asarray(img, dtype=np.uint8))
    return out


@dataclass(frozen=True)
class EvalStats:
    """Tally over N evaluation episodes."""

    source: str
    goals: int
    deaths: int
    timeouts: int
    returns: tuple[float, ...]

    @property
    def n_episodes(self) -> int:
        return self.goals + self.deaths + self.timeouts

    @property
    def total_points(self) -> float:
        return float(sum(self.returns))

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0.0

    @property
    def success_rate(self) -> float:
        n = self.n_episodes
        return self.goals / n if n else 0.0

    def summary(self) -> str:
        return (f"{self.source:>10}: goals {self.goals}  deaths {self.deaths}  "
                f"timeouts {self.timeouts}  | success {self.success_rate:.0%}  "
                f"points(total) {self.total_points:.1f}  mean_return {self.mean_return:.2f}")


def _classify(terminated: bool, truncated: bool, info: dict[str, Any]) -> str:
    if terminated and info.get("death"):
        return "death"
    if terminated:
        return "goal"
    return "timeout"


def run_episode(env: Any, action_fn: ActionFn, *, seed: int) -> tuple[str, float]:
    """One episode → (outcome, summed reward). Outcome ∈ {goal, death, timeout}."""
    obs, info = env.reset(seed=seed)
    ret = 0.0
    terminated = truncated = False
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(action_fn(env, obs))
        ret += float(reward)
        if terminated or truncated:
            break
    return _classify(terminated, truncated, info), ret


def evaluate(env: Any, action_fn: ActionFn, *, source: str, n_episodes: int = 20,
             seed0: int = 0) -> EvalStats:
    """Run ``n_episodes`` (distinct seeds) and tally goal/death/timeout + per-episode points."""
    outcomes: list[str] = []
    returns: list[float] = []
    for k in range(n_episodes):
        outcome, ret = run_episode(env, action_fn, seed=seed0 + k)
        outcomes.append(outcome)
        returns.append(ret)
    return EvalStats(source, outcomes.count("goal"), outcomes.count("death"),
                    outcomes.count("timeout"), tuple(returns))


def plot_scoreboard(stats: list[EvalStats], out_path: str | Path, *, title: str) -> Path:
    """Grouped bar chart (goals/deaths/timeouts per source) + a per-episode return strip → PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_bar, ax_ret) = plt.subplots(1, 2, figsize=(11, 4.2))
    cats = ["goal", "death", "timeout"]
    colors = {"goal": "#2e8b57", "death": "#c0392b", "timeout": "#7f8c8d"}
    width = 0.8 / max(1, len(stats))
    for i, s in enumerate(stats):
        vals = [s.goals, s.deaths, s.timeouts]
        x = np.arange(len(cats)) + (i - (len(stats) - 1) / 2) * width
        # spread alpha over [0.55, 1.0] across the series (was 0.6+0.4·i — >1 for a 3rd source).
        alpha = 0.55 + 0.45 * (i / max(1, len(stats) - 1))
        ax_bar.bar(x, vals, width, label=s.source,
                   color=[colors[c] for c in cats], edgecolor="black", alpha=alpha)
    ax_bar.set_xticks(np.arange(len(cats)))
    ax_bar.set_xticklabels(cats)
    ax_bar.set_ylabel("episodes")
    ax_bar.set_title("outcomes")
    ax_bar.legend(fontsize=8)
    for s in stats:
        ax_ret.plot(range(1, len(s.returns) + 1), s.returns, marker="o", ms=3, label=s.source)
    ax_ret.set_xlabel("episode")
    ax_ret.set_ylabel("points (return)")
    ax_ret.set_title("per-episode points")
    ax_ret.legend(fontsize=8)
    ax_ret.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# An overlay callback: given a per-frame context (step, return, action, + the step's info dict), return the
# HUD text lines to draw. ``None`` (default) = no overlay (unchanged behaviour).
OverlayFn = Callable[[dict[str, Any]], list[str]]


def _draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw HUD ``lines`` (top-left, on a dark box for legibility) onto an RGB frame."""
    from PIL import Image, ImageDraw, ImageFont
    if not lines:
        return frame
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)     # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    lh, pad = 16, 4
    w = max((int(draw.textlength(t, font=font)) for t in lines), default=0)
    draw.rectangle((1, 1, w + 2 * pad + 2, lh * len(lines) + pad), fill=(0, 0, 0))
    for i, t in enumerate(lines):
        draw.text((pad, 2 + i * lh), t, fill=(240, 240, 240), font=font)
    return np.asarray(img, dtype=np.uint8)


def render_episode_frames(env: Any, action_fn: ActionFn, *, seed: int = 0, width: int = 480,
                          height: int = 360, camera: Any = None, overlay: OverlayFn | None = None,
                          render_model: Any = None) -> list[np.ndarray]:
    """Roll one episode and return the rendered RGB frames (offscreen ``mujoco.Renderer``; needs a GL
    context). ``overlay`` draws a per-frame HUD from ``{"step","return","action",**info}``.

    ``render_model`` (optional) is a *visually decorated* ``MjModel`` (e.g. via ``decorate_scene`` — skybox +
    floor) with the **same DOF** as ``env.model``; the env drives the dynamics and the decorated model is
    synced to ``env.data`` each frame, so the scene renders with a sky instead of a black void."""
    import mujoco

    obs, info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    ret = 0.0
    rmodel = render_model if render_model is not None else env.model
    rdata = mujoco.MjData(rmodel) if render_model is not None else env.data
    renderer = mujoco.Renderer(rmodel, height=height, width=width)

    def grab(step: int, action: Any, info: dict[str, Any]) -> None:
        if render_model is not None:
            rdata.qpos[:] = env.data.qpos
            rdata.qvel[:] = env.data.qvel
            mujoco.mj_forward(rmodel, rdata)
        renderer.update_scene(rdata, camera=camera) if camera is not None \
            else renderer.update_scene(rdata)
        f = np.asarray(renderer.render(), dtype=np.uint8)
        if overlay is not None:
            f = _draw_overlay(f, overlay({"step": step, "return": ret, "action": action, **(info or {})}))
        frames.append(f)

    try:
        for step in range(env.max_steps):
            action = action_fn(env, obs)
            grab(step, action, info)
            obs, r, terminated, truncated, info = env.step(action)
            ret += float(r)
            if terminated or truncated:
                grab(step + 1, None, info)
                break
    finally:
        renderer.close()
    return frames


def _write_gif(frames: list[np.ndarray], out_path: str | Path, *, fps: int,
               stamp: str | None = None) -> Path:
    """Write ``frames`` to a GIF. ``stamp`` is a provenance label drawn bottom-right on every frame:
    ``None`` (default) auto-stamps the current local time; pass ``""`` to disable."""
    from PIL import Image
    label = now_stamp() if stamp is None else stamp
    frames = _stamp_frames(frames, label)
    out = Path(out_path).with_suffix(".gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=max(1, int(1000 / fps)), loop=0)
    return out


def render_episode_gif(env: Any, action_fn: ActionFn, out_path: str | Path, *, seed: int = 0,
                       width: int = 480, height: int = 360, fps: int = 20,
                       camera: Any = None, overlay: OverlayFn | None = None,
                       stamp: str | None = None) -> Path:
    """Render one episode to an animated GIF. ``overlay`` draws a per-frame HUD; ``stamp`` is a
    bottom-right provenance label (``None`` auto-stamps the time, ``""`` disables)."""
    frames = render_episode_frames(env, action_fn, seed=seed, width=width, height=height,
                                   camera=camera, overlay=overlay)
    return _write_gif(frames, out_path, fps=fps, stamp=stamp)


def compare_gif(panels: "list[list[np.ndarray]]", out_path: str | Path, *, fps: int = 30,
                stamp: str | None = None) -> Path:
    """Stitch several same-height frame-streams side-by-side into one comparison GIF.

    Streams of unequal length are padded by holding their last frame; a 4-px separator divides panels.
    ``stamp`` is a bottom-right provenance label (``None`` auto-stamps the time, ``""`` disables).
    # Preconditions all panels non-empty and the same frame height. # Postconditions one GIF written."""
    if not panels or not all(panels):
        raise ValueError("compare_gif needs non-empty panels")
    n = max(len(p) for p in panels)
    h = panels[0][0].shape[0]
    sep = np.full((h, 4, 3), 30, dtype=np.uint8)
    out_frames: list[np.ndarray] = []
    for t in range(n):
        row = []
        for p in panels:
            row.append(p[min(t, len(p) - 1)])
            row.append(sep)
        out_frames.append(np.hstack(row[:-1]))   # drop the trailing separator
    return _write_gif(out_frames, out_path, fps=fps, stamp=stamp)
