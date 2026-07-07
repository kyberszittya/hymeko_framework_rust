"""Campaign visual output — the graphical half of every experiment (CLAUDE.md §9 graphical-output strategy).

Three outputs per campaign: **numerical** (the caller's JSON/journal), **plotted** (:func:`plot_metric`), and
**animated** (:func:`render_actor_gif`). Rendering reuses :func:`hymeko_rl.eval.evaluate.render_episode_gif` and
plotting is a thin matplotlib layer — never re-implemented (§6.1). Higher-res defaults so the output is
slide-ready.
"""
from __future__ import annotations

import statistics as st
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.eval.evaluate import render_episode_gif


def _greedy_action_fn(actor: Any) -> Callable[[Any, np.ndarray], np.ndarray]:
    """The deterministic policy as an ``action_fn(env, obs) -> action`` (the form the renderer feeds)."""
    def fn(_env: Any, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = actor.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
        return np.asarray(a.squeeze(0).numpy(), dtype=np.float32)
    return fn


def render_actor_gif(env: Any, actor: Any, out_path: str | Path, *, seed: int = 20_000,
                     width: int = 960, height: int = 720, fps: int = 30,
                     camera: Any = None, overlay: Any = None) -> Path:
    """Render one greedy episode of ``actor`` on ``env`` to a high-res GIF (the **animated** output).

    # Preconditions ``actor`` exposes ``action_mean(Tensor) -> Tensor``; ``env`` is a gym 5-tuple MuJoCo env.
    # Postconditions one ``.gif`` written at ``>= 960x720``; reuses :func:`render_episode_gif`.
    """
    # MuJoCo's offscreen framebuffer defaults to 640x480; enlarge it on the model so the high-res render fits
    # (set before the Renderer is built). Idempotent — only ever grows it.
    g = env.model.vis.global_
    g.offwidth = max(int(width), int(g.offwidth))
    g.offheight = max(int(height), int(g.offheight))
    return render_episode_gif(env, _greedy_action_fn(actor), out_path, seed=seed,
                              width=width, height=height, fps=fps, camera=camera, overlay=overlay)


def plot_metric(records: Sequence[dict[str, Any]], out_path: str | Path, *, metric: str = "curve_max",
                title: str | None = None) -> Path:
    """Grouped-bar plot of ``metric`` per ``(task, algo)``, split by backbone (the **plotted** output).

    Each bar is the median over seeds; the HSiKAN-vs-MLP gap is read off the paired bars.
    # Preconditions each record has ``task``/``algo``/``backbone``/``metric`` keys; ``len(records) >= 1``.
    # Postconditions one ``.png`` written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    backbones: list[str] = []
    for r in records:
        if metric not in r:
            continue
        key = (str(r["task"]), str(r["algo"]))
        bb = str(r["backbone"])
        groups.setdefault(key, {}).setdefault(bb, []).append(float(r[metric]))
        if bb not in backbones:
            backbones.append(bb)
    if not groups:
        raise ValueError(f"no records carry metric {metric!r}")
    keys = sorted(groups)
    backbones = sorted(backbones)
    x = np.arange(len(keys), dtype=float)
    w = 0.8 / max(1, len(backbones))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(keys)), 4.0))
    for i, bb in enumerate(backbones):
        vals = [st.median(groups[k][bb]) if bb in groups[k] else float("nan") for k in keys]
        ax.bar(x + i * w, vals, w, label=bb)
    ax.set_xticks(x + w * (len(backbones) - 1) / 2.0)
    ax.set_xticklabels([f"{t}\n{a}" for t, a in keys], fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by task/algo (median over seeds)")
    ax.legend(title="backbone")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_curve_overlay(runs: "Sequence[tuple[str, dict[str, Any]]]", out_path: "str | Path", *,
                       metric: str = "delivery", seed_reduce: str = "median",
                       title: str | None = None) -> Path:
    """Overlay one ``metric``-vs-step curve per labelled run (each a campaign ``results.json`` dict) — the
    plotted comparison for A/Bs like *baseline-collapse vs framework-fix* (§9). Per run, the metric is reduced
    across seeds at each step (``median``/``mean``); a horizontal marker is drawn for the step-0 value (the BC
    floor) so a refine that merely holds the floor is visually distinct from one that climbs.

    # Preconditions each run dict has ``seeds: [{curve: [{step, <metric>}, …]}, …]``; ``len(runs) >= 1``.
    # Postconditions one ``.png`` (dpi 140) written; returns its path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("plot_curve_overlay needs at least one run")
    reduce = {"median": st.median, "mean": lambda xs: float(np.mean(xs))}[seed_reduce]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, res in runs:
        by_step: dict[float, list[float]] = {}
        for sd in res.get("seeds", []):
            for pt in sd.get("curve", []):
                if metric in pt:
                    by_step.setdefault(float(pt["step"]), []).append(float(pt[metric]))
        if not by_step:
            continue
        steps = sorted(by_step)
        vals = [reduce(by_step[s]) for s in steps]
        line, = ax.plot(steps, vals, marker="o", label=f"{label} (peak {max(vals):.2f})")
        ax.axhline(vals[0], color=line.get_color(), ls=":", alpha=0.5)   # the step-0 / BC floor
    ax.set_xlabel("environment steps")
    ax.set_ylabel(metric)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title or f"{metric} vs step (seed {seed_reduce})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
