"""Seminar figure: rotor vs transductive-table parameter efficiency.

Reads the multi-seed audit JSONL and writes a grouped bar chart showing that the
inductive Cayley-rotor embedding has a *constant* parameter count while the
transductive DADSGNN node table grows with the graph. AUROC (5-seed mean) is
annotated on each bar.

    python -m docs.seminar.make_rotor_param_figure   # or run directly
Out: docs/seminar/figures/rotor_param_efficiency.png  (deck untouched)
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "rotor_multiseed_20260616.jsonl"
BIG = ROOT / "reports" / "rotor_biggraph_20260616.jsonl"
OUT = ROOT / "docs" / "seminar" / "figures" / "rotor_param_efficiency.png"
OUT_PARETO = ROOT / "docs" / "seminar" / "figures" / "rotor_pareto.png"

DATASETS = ["bitcoin_alpha", "bitcoin_otc", "wiki_elec"]
MODELS = [
    ("cayley_rotor", "Cayley-rotor (inductive)", "#1b9e77"),
    ("dadsgnn", "DADSGNN (transductive table)", "#d95f02"),
]
# Full baseline family + colours for the big-graph Pareto plot.
PARETO_MODELS = [
    ("cayley_rotor", "Cayley-rotor", "#1b9e77", "*", 280),
    ("dadsgnn", "DADSGNN", "#d95f02", "o", 90),
    ("sgcn", "SGCN", "#7570b3", "s", 80),
    ("sigat", "SiGAT", "#e7298a", "^", 90),
]


def main() -> None:
    strict: dict[tuple, list[float]] = defaultdict(list)
    params: dict[tuple, int] = {}
    for line in SRC.read_text().splitlines():
        r = json.loads(line)
        if r["shuffle"]:
            continue
        k = (r["dataset"], r["model"])
        strict[k].append(r["test_auroc"])
        params[k] = r["n_params"]

    x = np.arange(len(DATASETS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (m, lab, col) in enumerate(MODELS):
        vals = [params[(d, m)] for d in DATASETS]
        aucs = [st.mean(strict[(d, m)]) for d in DATASETS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lab, color=col)
        for b, p, a in zip(bars, vals, aucs):
            ax.text(b.get_x() + b.get_width() / 2, p + 4000,
                    f"{p:,}\nAUROC {a:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("learned parameters")
    ax.set_ylim(0, max(params.values()) * 1.18)
    ax.set_title("Inductive rotor vs transductive table: parameters\n"
                 "constant 15,761 vs growing with the graph "
                 "(5 seeds, strict leakage protocol)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


def pareto() -> None:
    """Accuracy-vs-parameters Pareto scatter (log-x) on the large graphs --- the
    honest 'where does the rotor sit' picture: near-best AUROC at ~270x fewer
    parameters, but not the accuracy leader (SiGAT is)."""
    if not BIG.exists():
        print(f"skip pareto: {BIG} not found")
        return
    agg: dict[tuple, list[float]] = defaultdict(list)
    params: dict[tuple, int] = {}
    for line in BIG.read_text().splitlines():
        r = json.loads(line)
        if r["shuffle"]:
            continue
        k = (r["dataset"], r["model"])
        agg[k].append(r["test_auroc"])
        params[k] = r["n_params"]
    big_ds = ["epinions", "slashdot"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, ds in zip(axes, big_ds):
        for m, lab, col, mk, sz in PARETO_MODELS:
            k = (ds, m)
            if k not in params:
                continue
            ax.scatter(params[k], st.mean(agg[k]), s=sz, marker=mk,
                       color=col, edgecolor="black", linewidth=0.5,
                       label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel("learned parameters (log)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("strict AUROC (5-seed mean)")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Accuracy vs. parameters — the rotor is Pareto-efficient "
                 "(near-best AUROC at ~270x fewer params)")
    fig.tight_layout()
    fig.savefig(OUT_PARETO, dpi=160)
    print(f"wrote {OUT_PARETO}")


if __name__ == "__main__":
    main()
    pareto()
