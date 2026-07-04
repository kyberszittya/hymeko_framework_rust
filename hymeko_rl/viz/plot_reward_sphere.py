"""Plot the reward as a sphere — the alignment map over reward-shape space.

A reward ``R = wᵀf`` is scale-invariant in the weight vector ``w`` (the optimal policy is unchanged by
``w ↦ cw``), so a reward is a *direction* — a point on the unit sphere ``Sⁿ⁻¹``. For three chosen term axes
that is the ordinary 2-sphere, literally plottable. Each direction is handed to the reward-oracle
(:func:`hymeko_rl.eval.reward_oracle.certify`, ms-fast) which reports whether that reward's *optimum* delivers
or gets farmed. Colouring the sphere by that verdict draws the **reward-alignment frontier**: where the
delivering region begins.

Default axes (the farming ↔ delivering tension for galambos):
    x = in_zone               (the farmable per-step annuity — the trap)
    y = terminal_deliver      (the one-shot delivery event)
    z = conjunctive_progress  (the nonlinear farm-proof shaping)

    python -m hymeko_rl.viz.plot_reward_sphere            # writes reports/figures/reward_sphere.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hymeko_rl.env.reward import RewardSpec  # noqa: E402
from hymeko_rl.eval.reward_oracle import certify  # noqa: E402

_AXES = ("in_zone", "terminal_deliver", "conjunctive_progress")
_MAG = 200.0   # arbitrary (certify is scale-invariant); large enough that each direction is a meaningful reward


def _delivers(direction: "tuple[float, float, float]", axes: "tuple[str, str, str]") -> bool:
    """Certify the reward whose 3 term weights are ``_MAG · direction`` (drop ~zero-weight terms)."""
    terms = tuple((name, float(_MAG * w)) for name, w in zip(axes, direction) if w > 1e-6)
    if not terms:
        return False
    return certify(RewardSpec(terms)).delivers


def build(axes: "tuple[str, str, str]" = _AXES, n: int = 46) -> "dict[str, np.ndarray]":
    """Sample the positive-orthant of S² (all weights ≥ 0) and certify each direction.

    # Postconditions returns unit-sphere coords ``xyz`` (n²,3) and a bool mask ``delivers`` (n²,)."""
    th = np.linspace(0.0, np.pi / 2, n)      # polar   (z axis at 0)
    ph = np.linspace(0.0, np.pi / 2, n)      # azimuth (x↔y)
    T, P = np.meshgrid(th, ph)
    x = (np.sin(T) * np.cos(P)).ravel()
    y = (np.sin(T) * np.sin(P)).ravel()
    z = np.cos(T).ravel()
    xyz = np.stack([x, y, z], axis=1)
    delivers = np.array([_delivers((a, b, c), axes) for a, b, c in xyz], dtype=bool)
    return {"xyz": xyz, "delivers": delivers}


def plot(out: Path, axes: "tuple[str, str, str]" = _AXES, n: int = 46) -> Path:
    data = build(axes, n)
    xyz, dv = data["xyz"], data["delivers"]
    frac = float(dv.mean())

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    # a faint reference sphere octant
    u = np.linspace(0, np.pi / 2, 40)
    v = np.linspace(0, np.pi / 2, 40)
    ax.plot_surface(np.outer(np.sin(u), np.cos(v)), np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.cos(u), np.ones_like(v)), color="0.9", alpha=0.15, linewidth=0)
    ax.scatter(xyz[dv, 0], xyz[dv, 1], xyz[dv, 2], c="#2a9d3a", s=14, label=f"DELIVERS ({frac:.0%})", depthshade=True)
    ax.scatter(xyz[~dv, 0], xyz[~dv, 1], xyz[~dv, 2], c="#c0392b", s=14, label=f"farms ({1 - frac:.0%})", depthshade=True)
    # reference reward directions (unit-normalised) — where our variants sit
    refs = {
        "baseline (in_zone-heavy)": np.array([10.0, 3.0, 0.0]),
        "conj (terminal+conjprog)": np.array([0.0, 30.0, 40.0]),
        "pbrs-linear-ish": np.array([6.0, 30.0, 20.0]),
    }
    for lbl, w in refs.items():
        wn = w / np.linalg.norm(w)
        ax.scatter(*wn, c="black", s=90, marker="*", edgecolors="white", zorder=5)
        ax.text(wn[0], wn[1], wn[2] + 0.04, lbl, fontsize=8, ha="center")
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(axes[2])
    ax.set_title("Reward-alignment sphere — each point is a reward direction, oracle-certified\n"
                 f"(green = its optimum delivers; red = farms; delivering fraction {frac:.0%})", fontsize=10)
    ax.legend(loc="upper left")
    ax.view_init(elev=28, azim=38)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reports/figures/reward_sphere.png")
    ap.add_argument("--n", type=int, default=46)
    args = ap.parse_args(argv)
    p = plot(Path(args.out), n=args.n)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
