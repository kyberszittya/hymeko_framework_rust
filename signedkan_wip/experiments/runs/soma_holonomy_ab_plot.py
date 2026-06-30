"""Render the Gömb-Soma holonomy A/B bar chart from a train_mnist JSONL.

The chart contrasts test accuracy (5-seed mean ± pstd) of the three MNIST arms:
a linear control, the sign-as-routing base-Soma (the 2026-06-15 falsified
operator), and the sign-as-connection holonomy walk-conv. Data-shaping
(``summarize``) is a pure function so it is unit-testable without a display;
``render`` does the matplotlib draw — same split as ``bench_to_png`` (§6.1: no
re-implemented plotting machinery, a distinct schema warrants its own shaper).

Usage:
    python -m signedkan_wip.experiments.runs.soma_holonomy_ab_plot \
        --jsonl reports/soma_holonomy_vs_routing_mnist_20260629.jsonl \
        --out   reports/figures/soma_holonomy_ab_20260629.png
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

# Stable arm order (cheapest-structure → richest) + labels + deck palette.
ARM_ORDER = (
    "linear", "gomb_soma", "gomb_soma_holonomy",
    "gomb_soma_cheby", "gomb_soma_attn", "gomb_soma_posattn",
    "gomb_soma_tree", "gomb_soma_flat", "gomb_soma_cheby_flat",
)
ARM_LABEL = {
    "linear": "Linear control",
    "gomb_soma": "base-Soma\n(mean-pool)",
    "gomb_soma_holonomy": "Soma-holonomy\n(sign = connection)",
    "gomb_soma_cheby": "Cheby-CR cell\n(mean-pool)",
    "gomb_soma_attn": "attention pool\n(scale-free)",
    "gomb_soma_posattn": "pos-attention\n(what + where)",
    "gomb_soma_tree": "spatial tree\n(dynamic quadtree)",
    "gomb_soma_flat": "flatten\n(full spatial map)",
    "gomb_soma_cheby_flat": "Cheby-CR cell\n(flatten)",
}
ARM_COLOR = {
    "linear": "#bdbdbd",                   # grey: no structure
    "gomb_soma": "#8d6e63",                # brown: mean-pool (the bottleneck)
    "gomb_soma_holonomy": "#26a69a",       # green: the holonomy operator
    "gomb_soma_cheby": "#5c6bc0",          # indigo: expressivity axis
    "gomb_soma_attn": "#00897b",           # teal: scale-free attention pool
    "gomb_soma_posattn": "#3949ab",        # indigo: positional attention
    "gomb_soma_tree": "#00695c",           # deep teal: dynamic spatial tree
    "gomb_soma_flat": "#ef6c00",           # orange: full spatial map
    "gomb_soma_cheby_flat": "#8e24aa",     # purple: both axes
}


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape per-seed JSONL records into one bar per model arm.

    Preconditions
    -------------
    * ``records`` is a list of dicts each carrying ``model``, ``test_acc`` and
      ``n_params`` (the ``train_mnist`` JSONL schema).

    Postconditions
    -------------
    * One entry per arm present in ``records``, ordered by ``ARM_ORDER`` (unknown
      arms appended in first-seen order — never invented).
    * Each entry has ``mean``, ``pstd`` (population stdev; 0.0 for a single seed),
      ``n_seeds`` and ``n_params``.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_arm.setdefault(r["model"], []).append(r)
    order = [m for m in ARM_ORDER if m in by_arm]
    order += [m for m in by_arm if m not in ARM_ORDER]
    bars = []
    for arm in order:
        accs = [float(r["test_acc"]) for r in by_arm[arm]]
        bars.append(dict(
            model=arm,
            label=ARM_LABEL.get(arm, arm),
            mean=statistics.mean(accs),
            pstd=statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            n_seeds=len(accs),
            n_params=int(by_arm[arm][0]["n_params"]),
        ))
    return bars


_DEFAULT_TITLE = "Sign-as-routing vs sign-as-connection · Gömb-Soma walk-conv"
_DEFAULT_CAPTION = (
    "Same MNIST 5000/1000 split, 5 epochs, Adam 3e-3, CPU. "
    "base-Soma is the 2026-06-15 falsified operator; holonomy pools "
    "M_v(σ⊙m) with a single message bank."
)


def render(bars: list[dict[str, Any]], out_path: Any, *,
           title: str = _DEFAULT_TITLE, caption: str = _DEFAULT_CAPTION) -> Path:
    """Draw the accuracy bar chart (mean ± pstd) to ``out_path`` (PNG).

    ``title`` / ``caption`` default to the holonomy A/B strings (so that figure
    is unchanged); pass experiment-specific text for other sweeps.

    Preconditions: ``bars`` non-empty (from ``summarize``).
    Postconditions: a PNG is written at ``out_path``; returns its Path.
    """
    assert bars, "no bars to render"
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    xpos = list(range(len(bars)))
    means = [b["mean"] for b in bars]
    pstds = [b["pstd"] for b in bars]
    colors = [ARM_COLOR.get(b["model"], "#90a4ae") for b in bars]

    fig, ax = plt.subplots(figsize=(max(7.5, 2.0 * len(bars)), 5))
    ax.bar(xpos, means, yerr=pstds, color=colors, width=0.62,
           error_kw=dict(ecolor="#37474f", capsize=4))
    ax.set_xticks(xpos)
    ax.set_xticklabels([b["label"] for b in bars])
    ax.set_ylabel("MNIST test accuracy (5-seed mean ± pstd)")
    ax.set_ylim(0, 1.18)  # headroom so value labels clear the title
    ax.set_title(title, pad=12)
    for x, b in zip(xpos, bars):
        ax.text(x, b["mean"] + b["pstd"] + 0.02,
                f"{b['mean']:.3f}\n{b['n_params']}p",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.5, -0.13, caption,
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            color="#37474f")
    fig.tight_layout()
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def load_jsonl(path: Any) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of records (one dict per non-blank line)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def main(jsonl_path: Any = None, out_path: Any = None, *,
         title: str = _DEFAULT_TITLE, caption: str = _DEFAULT_CAPTION) -> Path:
    jsonl_path = Path(jsonl_path) if jsonl_path else Path(
        "reports/soma_holonomy_vs_routing_mnist_20260629.jsonl")
    out_path = Path(out_path) if out_path else Path(
        "reports/figures/soma_holonomy_ab_20260629.png")
    bars = summarize(load_jsonl(jsonl_path))
    if not bars:
        raise SystemExit(f"no records in {jsonl_path}")
    out = render(bars, out_path, title=title, caption=caption)
    print(f"Wrote {out}  ({len(bars)} arms)")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jsonl", default=None, help="train_mnist JSONL path")
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument("--title", default=_DEFAULT_TITLE)
    parser.add_argument("--caption", default=_DEFAULT_CAPTION)
    a = parser.parse_args()
    main(jsonl_path=a.jsonl, out_path=a.out, title=a.title, caption=a.caption)
