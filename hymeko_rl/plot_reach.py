"""Plot the reaching-error progression (for the simulation demo) + the BC loss curve.

Two panels: (left) behaviour-cloning MSE vs epoch; (right) mean end-effector error vs
env step for the closed-loop expert, the HSiKAN-cloned policy, and an untrained policy
(the floor). Headless (Agg); writes a PNG + PDF.

    python -m hymeko_rl.plot_reach --out reports/2026-06-18-hymeko-rl-phase1-reach
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hymeko_rl.bc import (  # noqa: E402
    _make_policy, behaviour_clone, collect_demos,
)
from hymeko_rl.env.arm_reach_env import ArmReachEnv  # noqa: E402
from hymeko_rl.policy import ActorCritic  # noqa: E402

ActionFn = Callable[[ArmReachEnv, np.ndarray], np.ndarray]


def _policy_fn(ac: ActorCritic) -> ActionFn:
    @torch.no_grad()
    def fn(_env: ArmReachEnv, obs: np.ndarray) -> np.ndarray:
        a = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
        return np.asarray(a.squeeze(0).numpy(), dtype=np.float32)
    return fn


def error_traces(env: ArmReachEnv, action_fn: ActionFn, n_ep: int, seed: int,
                 ) -> np.ndarray:
    """``(n_ep, max_steps)`` EE-error trajectories (NaN-padded for early-reached episodes)."""
    traces = np.full((n_ep, env.max_steps), np.nan)
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(env.max_steps):
            obs, _, _, truncated, info = env.step(action_fn(env, obs))
            traces[ep, t] = info["dist"]
            if truncated:
                break
    return traces


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="reports/2026-06-18-hymeko-rl-phase1-reach")
    ap.add_argument("--demos", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    env = ArmReachEnv()
    ac = _make_policy("hsikan", env, hidden=64)
    untrained = _make_policy("hsikan", env, hidden=64)
    obs, acts = collect_demos(env, a.demos, a.seed)
    losses = behaviour_clone(ac, obs, acts, n_epochs=a.epochs, seed=a.seed)

    arms: list[tuple[str, ActionFn, str]] = [
        ("expert (DLS-IK)", lambda e, _o: e.expert_action, "C2"),
        ("HSiKAN behaviour-cloned", _policy_fn(ac), "C0"),
        ("untrained (floor)", _policy_fn(untrained), "C3"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(losses, color="C0")
    ax1.set(yscale="log", xlabel="BC epoch", ylabel="MSE loss",
            title="Behaviour-cloning loss")
    ax1.grid(alpha=0.3)
    for label, fn, color in arms:
        tr = error_traces(ArmReachEnv(), fn, a.episodes, seed=5000)
        m, s = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0)
        x = np.arange(len(m))
        ax2.plot(x, m, label=label, color=color, linewidth=1.6)
        ax2.fill_between(x, m - s, m + s, alpha=0.15, color=color)
    ax2.set(xlabel="env step", ylabel="end-effector error (m)",
            title="Reaching error progression")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)
    fig.suptitle("HyMeKo arm reaching — Phase 1 (behaviour cloning over the kinematic "
                 "hypergraph)", y=1.0)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out.with_suffix('.png')} (final BC loss {losses[-1]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
