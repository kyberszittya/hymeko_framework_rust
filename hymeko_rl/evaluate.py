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
from pathlib import Path
from typing import Any

import numpy as np

ActionFn = Callable[[Any, np.ndarray], np.ndarray]


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
        ax_bar.bar(x, vals, width, label=s.source,
                   color=[colors[c] for c in cats], edgecolor="black", alpha=0.6 + 0.4 * i)
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


def render_episode_gif(env: Any, action_fn: ActionFn, out_path: str | Path, *, seed: int = 0,
                       width: int = 480, height: int = 360, fps: int = 20,
                       camera: Any = None) -> Path:
    """Render one episode to an animated GIF (offscreen ``mujoco.Renderer``; needs a GL context)."""
    import mujoco
    from PIL import Image

    obs, _info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    try:
        for _ in range(env.max_steps):
            renderer.update_scene(env.data, camera=camera) if camera is not None \
                else renderer.update_scene(env.data)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
            obs, _r, terminated, truncated, _info = env.step(action_fn(env, obs))
            if terminated or truncated:
                renderer.update_scene(env.data, camera=camera) if camera is not None \
                    else renderer.update_scene(env.data)
                frames.append(np.asarray(renderer.render(), dtype=np.uint8))
                break
    finally:
        renderer.close()
    out = Path(out_path).with_suffix(".gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=max(1, int(1000 / fps)), loop=0)
    return out
