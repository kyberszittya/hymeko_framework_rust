"""Demo 2 / plan-item 4 — forward-latency bench (the cost story, slide 18).

Reads the committed ``inference_bench.json`` (measured by
``experiments/runs/run_inference_bench.py``, ≥5 repeats after warm-up, CLAUDE.md
§3) and surfaces the **within-family lean-vs-joint** width gap with SGCN shown
for context. Renders the slide-18 horizontal bars per device. Pure reuse of
``bench_to_png.{bars_from_rows,render}`` — no measurement or plotting logic is
re-implemented here (CLAUDE.md §6.5 #2).

Inference-only: this demo *reads* the committed measurement; re-measuring is
``python -m signedkan_wip.experiments.runs.run_inference_bench`` (heavy, not a
room operation). The honest framing (report 2026-06-11): the defensible claims
are accuracy-per-parameter and the measured within-family width gap — **not**
"faster than SGCN".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..base import REPO_ROOT, DemoContext, DemoResult

DEFAULT_JSON = (
    REPO_ROOT / "signedkan_wip" / "experiments" / "results" / "inference_bench.json"
)
# The cells whose ratio is the slide-18 message.
LEAN_MODEL = "HSiKAN-lean"
JOINT_MODEL = "HSiKAN-joint"


def width_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Within-family lean→joint forward-latency ratio per (dataset, device).

    Preconditions: ``rows`` is the parsed ``inference_bench.json`` list; each
    HSiKAN row carries ``fwd_med_ms`` > 0.
    Postconditions: one record per (dataset, device) that has *both* the lean
    and joint cell; ``ratio = joint_med / lean_med``. Cells missing either
    variant are skipped (never fabricated).
    """
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["dataset"], r["device"], r["model"]): r for r in rows
    }
    keys = sorted({(r["dataset"], r["device"]) for r in rows})
    out: list[dict[str, Any]] = []
    for dataset, device in keys:
        lean = by_key.get((dataset, device, LEAN_MODEL))
        joint = by_key.get((dataset, device, JOINT_MODEL))
        if lean is None or joint is None:
            continue
        lean_ms = float(lean["fwd_med_ms"])
        joint_ms = float(joint["fwd_med_ms"])
        if lean_ms <= 0.0:
            continue
        out.append({
            "dataset": dataset,
            "device": device,
            "lean_ms": round(lean_ms, 2),
            "joint_ms": round(joint_ms, 2),
            "ratio": round(joint_ms / lean_ms, 2),
            "lean_params": int(lean.get("n_params", 0)),
            "joint_params": int(joint.get("n_params", 0)),
        })
    return out


class LatencyDemo:
    name = "latency"
    help = ("Forward-latency bench: within-family lean-vs-joint width gap "
            "(SGCN for context), slide-18 bars from the committed json.")

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--json", default=str(DEFAULT_JSON),
            help="path to inference_bench.json (committed measurement).",
        )
        parser.add_argument(
            "--no-figures", action="store_true",
            help="skip bar-chart PNG rendering (table only).",
        )

    def run(self, args: argparse.Namespace, ctx: DemoContext) -> DemoResult:
        os.environ.setdefault("MPLBACKEND", "Agg")
        json_path = Path(args.json)
        if not json_path.is_file():
            raise FileNotFoundError(
                f"inference_bench.json not found: {json_path}. Re-measure with "
                f"`python -m signedkan_wip.experiments.runs.run_inference_bench`."
            )
        json_rel = os.path.relpath(json_path, REPO_ROOT)
        rows = json.loads(json_path.read_text(encoding="utf-8"))

        result = DemoResult(name=self.name)
        ratios = width_ratios(rows)
        if not ratios:
            raise RuntimeError(
                f"{json_rel} has no (dataset, device) cell with both "
                f"{LEAN_MODEL!r} and {JOINT_MODEL!r}; nothing to report."
            )
        self._populate_metrics(result, ratios, json_rel)
        result.notes.append(
            "the lean(h=4)→joint(h=16) width gap is the message: "
            "~3.5x on CPU, ~1x on CUDA (same device, same graphs)."
        )
        result.notes.append(
            "the deck's 11x was the optuna_best_otc-vs-joint result — real but "
            "OTC-specific and tuple-set-driven, NOT a general width claim."
        )
        result.notes.append(
            "on absolute forward latency SGCN and MLP-blind are lightest; the "
            "defensible claims are accuracy-per-parameter and this within-family "
            "width gap — not 'faster than SGCN'.")

        if not args.no_figures:
            result.artifacts.extend(self._render(rows, ctx.out_dir))
        return result

    @staticmethod
    def _populate_metrics(
        result: DemoResult, ratios: list[dict[str, Any]], json_rel: str,
    ) -> None:
        """Per-cell lean/joint/ratio + per-device mean ratio, all provenanced
        to the committed json."""
        def put(key: str, value: Any) -> None:
            result.metrics[key] = value
            result.provenance[key] = json_rel

        for rec in ratios:
            tag = f"{rec['dataset']}/{rec['device']}"
            put(f"{tag} lean_ms", rec["lean_ms"])
            put(f"{tag} joint_ms", rec["joint_ms"])
            put(f"{tag} joint/lean", f"{rec['ratio']}x")
        # Headline ratios per device (mean across datasets), for the slide line.
        for device in dict.fromkeys(r["device"] for r in ratios):
            cells = [r for r in ratios if r["device"] == device]
            mean_ratio = sum(r["ratio"] for r in cells) / len(cells)
            put(f"mean joint/lean ({device})", f"{mean_ratio:.2f}x")

    @staticmethod
    def _render(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
        """Render slide-18 bars for every device present in the json."""
        from signedkan_wip.experiments.runs.bench_to_png import (
            bars_from_rows, render,
        )

        written: list[Path] = []
        for device in dict.fromkeys(r["device"] for r in rows):
            bars = bars_from_rows(rows, device)
            if not bars:
                continue
            written.append(render(bars, device, out_dir / f"latency_{device}.png"))
        return written


__all__ = ["LatencyDemo", "width_ratios"]
