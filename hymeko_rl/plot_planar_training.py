"""Plot the PPO training trace for the Galambos planar grasper.

Reads the metrics JSON written by ``train_planar_grasp`` (per-iteration return, policy/value loss,
entropy, approx-KL, clip fraction, action std, curriculum difficulty) and draws a multi-panel
training-curve figure.

    uv run python -m hymeko_rl.plot_planar_training --json checkpoints/galambos/ppo_trace.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (key, title, ylabel, smooth?) for each panel.
_PANELS = [
    ("return", "episodic return (reward)", "return", True),
    ("policy_loss", "policy (surrogate) loss", "loss", True),
    ("value_loss", "value (critic) loss", "loss", True),
    ("entropy", "policy entropy", "nats", True),
    ("approx_kl", "approx KL(old‖new)", "KL", True),
    ("clip_frac", "PPO clip fraction", "fraction", True),
    ("action_std", "action noise std  (exp log_std)", "std", False),
    ("difficulty", "curriculum difficulty", "0→1", False),
]


def _ema(ys: list[float], alpha: float = 0.1) -> list[float]:
    out: list[float] = []
    m = ys[0] if ys else 0.0
    for y in ys:
        m = alpha * y + (1 - alpha) * m
        out.append(m)
    return out


def plot(metrics: list[dict[str, float]], out_path: str | Path, *, title: str) -> Path:
    its = [m["iter"] for m in metrics]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for ax, (key, ttl, ylab, smooth) in zip(axes.flat, _PANELS):
        if not metrics or key not in metrics[0]:
            ax.axis("off")
            continue
        ys = [float(m.get(key, float("nan"))) for m in metrics]
        ax.plot(its, ys, color="#9DB4D6", lw=1.0, alpha=0.7, label="raw" if smooth else None)
        if smooth and len(ys) > 3:
            ax.plot(its, _ema(ys), color="#2563EB", lw=2.0, label="EMA")
        ax.set_title(ttl, fontsize=10, color="#201A40")
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if smooth and len(ys) > 3:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=13, color="#201A40", y=1.0)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", required=True, help="metrics JSON from train_planar_grasp")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside the JSON)")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.json).read_text())
    metrics = data.get("metrics", [])
    if not metrics:
        raise SystemExit(f"{a.json}: no per-iteration 'metrics' (rerun training with this build)")
    out = a.out or str(Path(a.json).with_name(Path(a.json).stem + "_curves"))
    p = plot(metrics, out, title=f"Galambos PPO training trace — {len(metrics)} iters "
             f"({data.get('strategy', '')})")
    print(f"wrote {p}  ({len(metrics)} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
