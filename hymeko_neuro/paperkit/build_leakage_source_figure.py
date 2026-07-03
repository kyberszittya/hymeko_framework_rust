"""Build the *leakage-by-source* figures for the Nature HSiKAN audit paper.

Two figures, both from measured label-shuffle JSONL (no new experiments):

**(1) Protocol lattice / mechanism** (``leakage_lattice``).
For the cycle/HSiKAN method (``cell_signed_graph`` full-graph cycle sigma-products),
plot the three reachability rules ``strict`` :math:`\\sqsubseteq` ``topo``
:math:`\\sqsubseteq` ``full``, each real vs shuffled-train-label. The shuffled bar
rises above chance **only** at ``full`` (the held-out sign is reachable in its own
cycle sigma-product); ``topo`` keeps the real signal high while the shuffled bar
collapses to chance --- i.e. *cycle topology is not leakage; the direct label-sign
channel is.* Data: ``cycle_reachability_grid.jsonl``.

**(2) Source contrast** (``leakage_source_contrast``).
Diffuse / "activated-neuron" readout baselines (node-embedding + MLP/attention
transformers) shown as real vs topo-shuffle --- structurally leak-free, the
shuffle bar sits at chance regardless of whether cycle *topology* is reachable ---
placed beside the local-readout cycle method, whose shuffle bar is at chance under
``topo`` but climbs under ``full``. The reader sees: activated-neuron
representations do not leak structurally; the local cycle-sign channel does, and
only when the label is reachable. Data: ``no_leak_baselines.jsonl`` (strict arm),
``no_leak_baselines_topo.jsonl`` (topo arm), ``cycle_reachability_grid.jsonl``
(cycle method).

Conventions match ``build_leakage_audit_figure.py``: matplotlib Agg, chance line
at 0.5, error bars (std over datasets x seeds), output pdf+png+eps to
``hymeko_neuro/assets/paper/figures/``.

Run::

  uv run python -m hymeko_neuro.paperkit.build_leakage_source_figure
  uv run python -m hymeko_neuro.paperkit.build_leakage_source_figure --self-check
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "hymeko_neuro" / "experiments" / "results"
FIG_DIR = REPO / "hymeko_neuro" / "paper" / "figures"

GRID = RESULTS / "cycle_reachability_grid.jsonl"
BASE_STRICT = RESULTS / "no_leak_baselines.jsonl"
BASE_TOPO = RESULTS / "no_leak_baselines_topo.jsonl"

CHANCE = 0.5
LEAK_THRESHOLD = 0.55  # AUROC above this under shuffle = residual leak (paper convention)

RULE_ORDER = ["strict", "topo", "full"]
RULE_LABEL = {
    "strict": "strict\n(all test cycles excluded)",
    "topo": "topo\n(topology kept, sign masked)",
    "full": "full\n(test sign reachable)",
}
# Compact labels for the narrow right panel of the source-contrast figure.
RULE_LABEL_SHORT = {"strict": "strict", "topo": "topo\n(sign masked)", "full": "full\n(sign reachable)"}

# Diffuse / "activated-neuron" readout baselines (node embeddings + MLP/attention).
BASELINE_ORDER = ["sgcn", "sigat", "sgt", "sgcl", "sigformer", "sesgformer", "dadsgnn"]
BASELINE_LABEL = {
    "sgcn": "SGCN", "sigat": "SiGAT", "sgt": "SGT", "sgcl": "SGCL",
    "sigformer": "SiGformer", "sesgformer": "SE-SGformer", "dadsgnn": "DADSGNN",
}

COL_REAL = "#1b6ca8"
COL_SHUF = "#9aa0a6"
COL_CHANCE = "#b02a2a"
COL_LEAK = "#d98a00"


@dataclass(frozen=True)
class Arm:
    """Mean +/- std of a real/shuffle arm over its pooled cells."""

    mean: float
    std: float
    n: int

    @classmethod
    def from_values(cls, xs: list[float]) -> "Arm | None":
        xs = [x for x in xs if x is not None and not math.isnan(x)]
        if not xs:
            return None
        return cls(float(np.mean(xs)), float(np.std(xs)), len(xs))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_lattice(path: Path) -> dict[str, dict[bool, Arm]]:
    """``{rule: {shuffle: Arm}}`` for the cycle method, pooled over datasets+seeds.

    Preconditions: rows carry ``rule``, ``shuffle``, ``auc``; ``(dataset, rule,
    seed, shuffle)`` identifies a cell. Duplicate cells (re-runs) are deduped,
    last-wins, so no seed is double-weighted.
    """
    seen: dict[tuple, float] = {}
    for r in _read_jsonl(path):
        key = (r.get("dataset"), r["rule"], r.get("seed"), bool(r["shuffle"]))
        seen[key] = r.get("auc")
    pooled: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {False: [], True: []})
    for (_ds, rule, _seed, shuf), auc in seen.items():
        pooled[rule][shuf].append(auc)
    out: dict[str, dict[bool, Arm]] = {}
    for rule, arms in pooled.items():
        real = Arm.from_values(arms[False])
        shuf = Arm.from_values(arms[True])
        if real is not None and shuf is not None:
            out[rule] = {False: real, True: shuf}
    return out


def load_baselines(path: Path) -> dict[str, dict[bool, Arm]]:
    """``{model: {shuffle: Arm}}`` pooled over datasets+seeds, deduped last-wins.

    Cell key ``(model, dataset, seed, shuffle)``; the topo file has duplicate
    seed-0 re-runs that this dedup folds to one value each.
    """
    seen: dict[tuple, float] = {}
    for r in _read_jsonl(path):
        key = (r["model"], r.get("dataset"), r.get("seed"), bool(r["shuffle"]))
        seen[key] = r.get("auc")
    pooled: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {False: [], True: []})
    for (model, _ds, _seed, shuf), auc in seen.items():
        pooled[model][shuf].append(auc)
    out: dict[str, dict[bool, Arm]] = {}
    for model, arms in pooled.items():
        real = Arm.from_values(arms[False])
        shuf = Arm.from_values(arms[True])
        if real is not None and shuf is not None:
            out[model] = {False: real, True: shuf}
    return out


def _grouped_bars(
    ax,
    groups: list[str],
    real: list[Arm],
    shuf: list[Arm],
    shuf_label: str,
) -> None:
    """Real/shuffle paired bars with std error bars, chance + leak guide lines."""
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w / 2, [a.mean for a in real], w, yerr=[a.std for a in real],
           capsize=3, color=COL_REAL, label="real labels")
    ax.bar(x + w / 2, [a.mean for a in shuf], w, yerr=[a.std for a in shuf],
           capsize=3, color=COL_SHUF, label=shuf_label)
    ax.axhline(CHANCE, ls="--", lw=1.2, color=COL_CHANCE, label="chance (0.5)")
    ax.axhline(LEAK_THRESHOLD, ls=":", lw=1.1, color=COL_LEAK, label="leak threshold (0.55)")
    ax.set_xticks(x)
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis="y", ls=":", color="#aaa", lw=0.5, alpha=0.5)


def build_lattice_figure(lattice: dict[str, dict[bool, Arm]], out_stem: str) -> int:
    """Strict/topo/full real-vs-shuffle for the cycle method. Returns #rules drawn."""
    rules = [r for r in RULE_ORDER if r in lattice]
    if not rules:
        raise SystemExit("lattice data has no rule with both real+shuffle arms")
    real = [lattice[r][False] for r in rules]
    shuf = [lattice[r][True] for r in rules]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    _grouped_bars(ax, rules, real, shuf, "shuffled train labels")
    ax.set_xticklabels([RULE_LABEL[r] for r in rules], fontsize=9)
    ax.set_title("Cycle/HSiKAN method: leakage appears only at $R_{\\mathrm{full}}$\n"
                 "(topology is a clean feature; the label-sign channel is the leak)")
    # annotate each shuffle bar with its value, above the error bar, clear of guides
    for i, a in enumerate(shuf):
        ax.annotate(f"{a.mean:.3f}", (i + 0.19, a.mean + a.std + 0.018),
                    ha="center", fontsize=8, color="#333", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    _save(fig, out_stem)
    return len(rules)


def build_source_contrast_figure(
    baselines_strict: dict[str, dict[bool, Arm]],
    baselines_topo: dict[str, dict[bool, Arm]],
    lattice: dict[str, dict[bool, Arm]],
    out_stem: str,
) -> int:
    """Diffuse-readout baselines (real vs topo-shuffle) beside the cycle method.

    Left panel: each baseline's shuffle bar under the strict arm and the topo
    arm sit together at chance -> no structural leak from activated-neuron
    readout. Right panel: the cycle method (local readout) -- topo-shuffle at
    chance, full-shuffle climbing above the leak line. Returns #baseline groups.
    """
    models = [m for m in BASELINE_ORDER if m in baselines_strict and m in baselines_topo]
    if not models:
        raise SystemExit("no baseline model present in both strict and topo files")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.5, 4.3), dpi=120, gridspec_kw={"width_ratios": [len(models), 2.1]}
    )

    # --- Left: diffuse-readout baselines, real / strict-shuffle / topo-shuffle ---
    x = np.arange(len(models))
    w = 0.27
    real = [baselines_strict[m][False] for m in models]
    shuf_s = [baselines_strict[m][True] for m in models]
    shuf_t = [baselines_topo[m][True] for m in models]
    axL.bar(x - w, [a.mean for a in real], w, yerr=[a.std for a in real],
            capsize=2, color=COL_REAL, label="real labels")
    axL.bar(x, [a.mean for a in shuf_s], w, yerr=[a.std for a in shuf_s],
            capsize=2, color=COL_SHUF, label="shuffle (strict)")
    axL.bar(x + w, [a.mean for a in shuf_t], w, yerr=[a.std for a in shuf_t],
            capsize=2, color="#c4c8cc", hatch="//", edgecolor="#7a7f85",
            label="shuffle (topo)")
    axL.axhline(CHANCE, ls="--", lw=1.2, color=COL_CHANCE)
    axL.axhline(LEAK_THRESHOLD, ls=":", lw=1.1, color=COL_LEAK)
    axL.set_xticks(x)
    axL.set_xticklabels([BASELINE_LABEL[m] for m in models], rotation=20, ha="right", fontsize=9)
    axL.set_ylabel("held-out AUROC")
    axL.set_ylim(0.4, 1.0)
    axL.grid(True, axis="y", ls=":", color="#aaa", lw=0.5, alpha=0.5)
    axL.set_title("Diffuse / activated-neuron readout (baselines):\n"
                  "shuffle at chance under strict AND topo -> no structural leak")
    axL.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=2)

    # --- Right: local-readout cycle method, topo vs full ---
    rules = [r for r in ("topo", "full") if r in lattice]
    xr = np.arange(len(rules))
    wr = 0.38
    creal = [lattice[r][False] for r in rules]
    cshuf = [lattice[r][True] for r in rules]
    axR.bar(xr - wr / 2, [a.mean for a in creal], wr, yerr=[a.std for a in creal],
            capsize=3, color=COL_REAL, label="real labels")
    axR.bar(xr + wr / 2, [a.mean for a in cshuf], wr, yerr=[a.std for a in cshuf],
            capsize=3, color=COL_SHUF, label="shuffled train labels")
    axR.axhline(CHANCE, ls="--", lw=1.2, color=COL_CHANCE)
    axR.axhline(LEAK_THRESHOLD, ls=":", lw=1.1, color=COL_LEAK)
    axR.set_xticks(xr)
    axR.set_xticklabels([RULE_LABEL_SHORT[r] for r in rules], fontsize=9)
    axR.set_ylim(0.4, 1.0)
    axR.grid(True, axis="y", ls=":", color="#aaa", lw=0.5, alpha=0.5)
    axR.set_title("Local readout (cycle/HSiKAN method):\n"
                  "clean at topo, leaks at full")
    for i, a in enumerate(cshuf):
        axR.annotate(f"{a.mean:.3f}", (i + 0.19, a.mean + a.std + 0.018),
                     ha="center", fontsize=8, color="#333", fontweight="bold")
    axR.legend(loc="upper right", fontsize=8, framealpha=0.95)

    fig.suptitle("Leakage by source: activated-neuron representations are structurally "
                 "leak-free; the local cycle-sign channel is where leakage concentrates",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_stem)
    return len(models)


def _save(fig, out_stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "eps"):
        out = FIG_DIR / f"{out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=140 if ext == "png" else None)
    plt.close(fig)


def _synthetic_lattice() -> dict[str, dict[bool, Arm]]:
    return {
        "strict": {False: Arm(0.500, 0.0, 10), True: Arm(0.500, 0.0, 10)},
        "topo": {False: Arm(0.82, 0.02, 10), True: Arm(0.48, 0.02, 10)},
        "full": {False: Arm(0.88, 0.02, 10), True: Arm(0.70, 0.04, 10)},
    }


def _synthetic_baselines(shuf_mean: float) -> dict[str, dict[bool, Arm]]:
    return {
        m: {False: Arm(0.87, 0.02, 25), True: Arm(shuf_mean, 0.01, 25)}
        for m in BASELINE_ORDER
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, default=GRID)
    ap.add_argument("--base-strict", type=Path, default=BASE_STRICT)
    ap.add_argument("--base-topo", type=Path, default=BASE_TOPO)
    ap.add_argument("--lattice-stem", default="leakage_lattice")
    ap.add_argument("--source-stem", default="leakage_source_contrast")
    ap.add_argument("--self-check", action="store_true",
                    help="build both figures from deterministic synthetic data (no I/O deps)")
    args = ap.parse_args()

    if args.self_check:
        lat = _synthetic_lattice()
        bs = _synthetic_baselines(0.52)
        bt = _synthetic_baselines(0.51)
    else:
        for p in (args.grid, args.base_strict, args.base_topo):
            if not p.exists():
                raise SystemExit(f"missing data file: {p}")
        lat = load_lattice(args.grid)
        bs = load_baselines(args.base_strict)
        bt = load_baselines(args.base_topo)

    n_rules = build_lattice_figure(lat, args.lattice_stem)
    n_models = build_source_contrast_figure(bs, bt, lat, args.source_stem)
    print(f"wrote {FIG_DIR / args.lattice_stem}.{{pdf,png,eps}}  ({n_rules} rules)")
    print(f"wrote {FIG_DIR / args.source_stem}.{{pdf,png,eps}}  ({n_models} baselines + cycle method)")


if __name__ == "__main__":
    main()
