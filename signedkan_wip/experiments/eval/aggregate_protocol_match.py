"""Aggregate the protocol-matched comparison JSONL into the honest gap table.

Consumes the rows written by
:mod:`signedkan_wip.experiments.runs.run_protocol_matched_sigat` and emits, per
dataset, a model × protocol table of held-out AUROC (mean ± std over seeds) with
the shuffle-arm mean as the leakage gate, plus each model's gap versus the
reference line. The point of the plan: the headline "~0.04 SiGAT gap" was
measured cross-protocol (target non-deduped, our line deduped); here every cell
is like-for-like, so the gap shown is honest.

Pure stdlib (no torch) — a light import for the reporting step.

Run:
    python -m signedkan_wip.experiments.eval.aggregate_protocol_match \
        --in signedkan_wip/experiments/results/protocol_matched_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GATE = 0.55                     # shuffled AUROC at/below this is "clean" (chance)
REFERENCE_MODEL = "hsikan_rotor_r2sw4"  # gaps are reported relative to our line


def _finite(values: list[float | None]) -> list[float]:
    """Keep only finite floats (drop ``None`` and ``nan`` — degenerate slices)."""
    return [v for v in values if v is not None and not math.isnan(float(v))]


@dataclass(frozen=True)
class CellStat:
    """Aggregated held-out stats for one (model, dataset, protocol) over seeds."""
    model: str
    dataset: str
    dedup: bool
    n_seeds: int          # seeds with a finite real AUROC
    real_mean: float
    real_std: float
    shuf_mean: float      # leakage gate (nan if no finite shuffle arm)

    @property
    def clean(self) -> bool:
        return not math.isnan(self.shuf_mean) and self.shuf_mean <= GATE


def aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str, bool], CellStat]:
    """Group rows by (model, dataset, dedup) and reduce real/shuffle arms.

    Precondition: each row carries ``model``, ``dataset``, ``dedup``, ``shuffle``,
    ``auc``. Postcondition: one :class:`CellStat` per present group.
    """
    real: dict[tuple[str, str, bool], list[float | None]] = defaultdict(list)
    shuf: dict[tuple[str, str, bool], list[float | None]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["dataset"], bool(r["dedup"]))
        (shuf if r["shuffle"] else real)[key].append(r.get("auc"))

    out: dict[tuple[str, str, bool], CellStat] = {}
    for key, reals in real.items():
        fr = _finite(reals)
        fs = _finite(shuf.get(key, []))
        out[key] = CellStat(
            model=key[0], dataset=key[1], dedup=key[2],
            n_seeds=len(fr),
            real_mean=stats.mean(fr) if fr else float("nan"),
            real_std=stats.pstdev(fr) if len(fr) > 1 else 0.0,
            shuf_mean=stats.mean(fs) if fs else float("nan"),
        )
    return out


def _fmt(stat: CellStat | None) -> str:
    if stat is None or math.isnan(stat.real_mean):
        return "—"
    gate = "✓" if stat.clean else "⚠"
    return f"{stat.real_mean:.4f}±{stat.real_std:.4f} (g{stat.shuf_mean:.2f}{gate})"


def render(table: dict[tuple[str, str, bool], CellStat]) -> str:
    """Markdown: one block per dataset, models × {non-deduped, deduped} + gap."""
    datasets = sorted({k[1] for k in table})
    models = sorted({k[0] for k in table})
    lines: list[str] = []
    for ds in datasets:
        lines.append(f"\n### {ds}\n")
        lines.append("| model | non-deduped | deduped | Δ vs ref (raw / dedup) |")
        lines.append("|---|---|---|---|")
        ref_raw = table.get((REFERENCE_MODEL, ds, False))
        ref_ded = table.get((REFERENCE_MODEL, ds, True))
        for m in models:
            raw = table.get((m, ds, False))
            ded = table.get((m, ds, True))
            d_raw = _gap(raw, ref_raw)
            d_ded = _gap(ded, ref_ded)
            star = "  ← ref" if m == REFERENCE_MODEL else ""
            lines.append(f"| {m}{star} | {_fmt(raw)} | {_fmt(ded)} | {d_raw} / {d_ded} |")
    lines.append(f"\n_Gate: g = mean shuffled AUROC; ✓ ≤ {GATE} (clean), "
                 f"⚠ > {GATE} (reads label through structure — not reportable). "
                 f"Δ = model − {REFERENCE_MODEL} (positive ⇒ ahead of our line)._")
    return "\n".join(lines)


def _gap(stat: CellStat | None, ref: CellStat | None) -> str:
    if (stat is None or ref is None
            or math.isnan(stat.real_mean) or math.isnan(ref.real_mean)):
        return "—"
    return f"{stat.real_mean - ref.real_mean:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="protocol-matched JSONL written by run_protocol_matched_sigat")
    a = ap.parse_args()
    # The table carries ±/Δ/✓ glyphs; force UTF-8 so a cp1250 Windows console
    # (or any non-UTF-8 locale) prints them instead of raising UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = Path(a.in_path)
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    table = aggregate(rows)
    print(render(table))


if __name__ == "__main__":
    main()
