"""Render the slide-18 horizontal-bar chart from ``inference_bench.json``.

The chart contrasts SGCN, HSiKAN-lean, and HSiKAN-joint forward latency per
(dataset, device) cell. Data-shaping (``bars_from_rows``) is a pure function so
it is unit-testable without a display; ``render`` does the matplotlib draw.

Honesty (INFERENCE_DEMOS Demo 2): the figure is titled to make the within-family
lean-vs-joint gap the message, with SGCN shown for context — not a "faster than
SGCN" claim.

Usage:
    python -m hymeko_neuro.experiments.runs.bench_to_png
    python -m hymeko_neuro.experiments.runs.bench_to_png --device cpu --out fig.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Stable model order + colour map (purple-blue-green deck palette, §SEMINAR).
MODEL_ORDER = ("SGCN", "SiGAT", "SGT", "MLP-blind",
               "HSiKAN-lean", "HSiKAN-joint")
MODEL_COLOR = {
    "SGCN": "#7e57c2",          # purple
    "SiGAT": "#5c6bc0",         # indigo
    "SGT": "#8d6e63",           # brown
    "MLP-blind": "#bdbdbd",     # grey (no structure)
    "HSiKAN-lean": "#26a69a",   # green
    "HSiKAN-joint": "#42a5f5",  # blue
}


def bars_from_rows(rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    """Shape bench rows into ordered horizontal bars for one device.

    Returns a list of ``dict(label, model, fwd_med_ms, fwd_iqr_ms,
    fwd_worst_ms)`` grouped by dataset, each dataset's models in MODEL_ORDER,
    top-to-bottom in the order datasets first appear.

    Preconditions: ``rows`` is the parsed json list; ``device`` in {cpu, cuda}.
    Postconditions: every returned bar has a positive ``fwd_med_ms``; unknown
    models are skipped (not invented).
    """
    cells = [r for r in rows if r.get("device") == device]
    datasets = list(dict.fromkeys(r["dataset"] for r in cells))
    by_key = {(r["dataset"], r["model"]): r for r in cells}
    bars = []
    for dataset in datasets:
        for model in MODEL_ORDER:
            r = by_key.get((dataset, model))
            if r is None:
                continue
            bars.append(dict(
                label=f"{dataset} · {model}",
                dataset=dataset, model=model,
                fwd_med_ms=float(r["fwd_med_ms"]),
                fwd_iqr_ms=float(r.get("fwd_iqr_ms", 0.0)),
                fwd_worst_ms=float(r.get("fwd_worst_ms", r["fwd_med_ms"])),
            ))
    return bars


def render(bars: list[dict[str, Any]], device: str, out_path: Any) -> Path:
    """Draw the horizontal-bar chart to ``out_path`` (PNG).

    Preconditions: ``bars`` non-empty (from ``bars_from_rows``).
    Postconditions: a PNG is written at ``out_path``; returns its Path.
    """
    assert bars, "no bars to render"
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    labels = [b["label"] for b in bars]
    meds = [b["fwd_med_ms"] for b in bars]
    iqrs = [b["fwd_iqr_ms"] for b in bars]
    colors = [MODEL_COLOR.get(b["model"], "#bdbdbd") for b in bars]
    ypos = list(range(len(bars)))

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(bars) + 1.5))
    ax.barh(ypos, meds, xerr=iqrs, color=colors, height=0.7,
            error_kw=dict(ecolor="#37474f", capsize=3))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first dataset on top
    ax.set_xlabel("forward latency (ms, median ± IQR)")
    ax.set_title(f"Forward latency · {device.upper()} — within-family "
                 "lean vs joint (SGCN for context)")
    for y, m in zip(ypos, meds):
        ax.text(m, y, f" {m:.1f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


# --------------------------------------------------------------------------- #
# Accuracy-vs-params Pareto. Cost axis is params (config-intrinsic, committed),  #
# so tuned (optuna) and untuned (phase8) points never share a latency axis.      #
# --------------------------------------------------------------------------- #
def pareto_points(records: dict[Any, Any], dataset: str) -> list[Any]:
    """AccuracyRecords for one dataset, ascending by params (cheapest first).

    Preconditions: ``records`` is the (dataset, method)→AccuracyRecord map.
    Postconditions: only this dataset's records; never fabricates a point.
    """
    pts = [r for (ds, _m), r in records.items() if ds == dataset]
    return sorted(pts, key=lambda r: r.n_params)


def render_pareto(points: list[Any], dataset: str, out_path: Any) -> Path:
    """Scatter: y = mean AUC, x = params (log). Tuned points are starred.

    Preconditions: ``points`` non-empty.
    Postconditions: a PNG is written at ``out_path``; returns its Path.
    """
    assert points, "no pareto points"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for p in points:
        tuned = getattr(p, "tuned", False)
        is_hsikan = "HSiKAN" in p.method
        ax.errorbar(p.n_params, p.mean_auc, yerr=getattr(p, "std_auc", 0.0),
                    fmt="*" if tuned else ("D" if is_hsikan else "o"),
                    ms=15 if tuned else (9 if is_hsikan else 7),
                    color="#d81b60" if is_hsikan else "#455a64",
                    ecolor="#90a4ae", capsize=3, zorder=3)
        ax.annotate(f" {p.method}", (p.n_params, p.mean_auc), fontsize=8,
                    va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale) — 'fraction of the cost'")
    ax.set_ylabel("test ROC-AUC (5-seed mean ± pstdev)")
    ax.set_title(f"Accuracy vs cost · {dataset}")
    ax.grid(True, which="both", alpha=0.25)
    # Honesty caption: baselines untuned, the starred HSiKAN point is tuned.
    ax.text(0.5, -0.16,
            "★ = optuna-tuned HSiKAN · ◆ = HSiKAN (untuned panel) · "
            "● = untuned baselines.\nTuning applied to HSiKAN only; baselines "
            "are the apples-to-apples phase8 panel.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            color="#37474f")
    fig.tight_layout()
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def table_rows(latency_rows: list[dict[str, Any]], records: dict[Any, Any],
               dataset: str) -> list[dict[str, Any]]:
    """Merge here-measured latency with committed accuracy into one table.

    Union of methods across both sources; a missing axis is ``None`` (rendered
    as '—'), never faked. Sorted by params ascending.
    """
    lat = {r["model"]: r for r in latency_rows if r["dataset"] == dataset}
    lat_cuda = {(r["model"]): r for r in latency_rows
                if r["dataset"] == dataset and r["device"] == "cuda"}
    lat_cpu = {(r["model"]): r for r in latency_rows
               if r["dataset"] == dataset and r["device"] == "cpu"}
    acc = {m: r for (ds, m), r in records.items() if ds == dataset}
    methods = list(dict.fromkeys(list(acc) + list(lat)))
    out = []
    for m in methods:
        a = acc.get(m)
        cpu, cuda = lat_cpu.get(m), lat_cuda.get(m)
        params = a.n_params if a else (cpu or cuda or {}).get("n_params")
        out.append(dict(
            method=m,
            mean_auc=a.mean_auc if a else None,
            std_auc=a.std_auc if a else None,
            n_params=params,
            fwd_cpu_ms=cpu["fwd_med_ms"] if cpu else None,
            fwd_cuda_ms=cuda["fwd_med_ms"] if cuda else None,
            tuned=a.tuned if a else None,
            source=a.source if a else "this-bench (latency only)",
        ))
    return sorted(out, key=lambda r: (r["n_params"] is None, r["n_params"] or 0))


def write_table(latency_rows: list[dict[str, Any]], records: dict[Any, Any],
                dataset: str, out_path: Any) -> Path:
    """Write the merged accuracy+latency markdown table for one dataset."""
    def fmt(x: Any, spec: str = ".4f") -> str:
        return "—" if x is None else format(x, spec)

    lines = [f"### {dataset}", "",
             "| method | AUC (5-seed) | params | fwd CPU ms | fwd CUDA ms | tuned | source |",
             "|---|---:|---:|---:|---:|:--:|---|"]
    for r in table_rows(latency_rows, records, dataset):
        auc = "—" if r["mean_auc"] is None else f"{r['mean_auc']:.4f} ± {r['std_auc']:.4f}"
        tuned = "—" if r["tuned"] is None else ("yes" if r["tuned"] else "no")
        lines.append(
            f"| {r['method']} | {auc} | {fmt(r['n_params'], 'd') if r['n_params'] else '—'} "
            f"| {fmt(r['fwd_cpu_ms'], '.2f')} | {fmt(r['fwd_cuda_ms'], '.2f')} "
            f"| {tuned} | {r['source']} |")
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def main(json_path: Any = None, device: str = "cpu", out_path: Any = None) -> Path:
    json_path = Path(json_path) if json_path else Path(
        "hymeko_neuro/experiments/results/inference_bench.json")
    out_path = Path(out_path) if out_path else Path(
        f"hymeko_neuro/experiments/results/inference_bench_{device}.png")
    rows = json.loads(json_path.read_text())
    bars = bars_from_rows(rows, device)
    if not bars:
        raise SystemExit(f"no rows for device={device} in {json_path}")
    out = render(bars, device, out_path)
    print(f"Wrote {out}  ({len(bars)} bars)")
    return out


def main_all(json_path: Any = None, results_dir: Any = None) -> list[Path]:
    """Render every slide-18 artifact: latency bars (both devices), accuracy
    Pareto (both datasets) and the merged numbers table."""
    from hymeko_neuro.experiments.runs.sota_compare import (
        load_optuna_best, load_phase8_accuracy, merge_records,
    )

    rdir = Path(results_dir) if results_dir else Path(
        "hymeko_neuro/experiments/results")
    json_path = Path(json_path) if json_path else rdir / "inference_bench.json"
    rows = json.loads(json_path.read_text())
    records = merge_records(
        load_phase8_accuracy(rdir / "phase8_bitcoin_5seed.json"),
        load_optuna_best(rdir / "bitcoin_optuna_best_5seed_2026_05_13.jsonl"),
    )
    written: list[Path] = []
    for device in ("cpu", "cuda"):
        bars = bars_from_rows(rows, device)
        if bars:
            written.append(render(bars, device, rdir / f"inference_bench_{device}.png"))
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        pts = pareto_points(records, dataset)
        if pts:
            written.append(render_pareto(
                pts, dataset, rdir / f"inference_bench_pareto_{dataset}.png"))
        written.append(write_table(
            rows, records, dataset, rdir / f"inference_bench_table_{dataset}.md"))
    for w in written:
        print(f"Wrote {w}")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="render slide-18 latency bars")
    parser.add_argument("--json", default=None, help="inference_bench.json path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument("--all", action="store_true",
                        help="render bars + Pareto + table for all cells")
    a = parser.parse_args()
    if a.all:
        main_all(json_path=a.json)
    else:
        main(json_path=a.json, device=a.device, out_path=a.out)
