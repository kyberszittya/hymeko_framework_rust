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
