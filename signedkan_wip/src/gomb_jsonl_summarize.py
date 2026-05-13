"""Summarize ``run_gomb_tune`` JSONL files: one Markdown table row per phase.

Reads all lines; keeps objects with ``"tuner_phase_summary": true``;
prints dataset, ``best_score``, ``tuner_pick_best_by``, ``best_test_auroc``,
``best_val_auroc``, ``best_n_params``, wall seconds.

Example::

    python -m signedkan_wip.src.gomb_jsonl_summarize reports/foo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "jsonl",
        type=Path,
        help="Path to JSONL written by run_gomb_tune.",
    )
    args = ap.parse_args()
    raw = args.jsonl.read_text(encoding="utf-8")
    summaries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("tuner_phase_summary") is True:
            summaries.append(obj)

    if not summaries:
        print("(no tuner_phase_summary rows found)", file=sys.stderr)
        sys.exit(1)

    cols = (
        "dataset",
        "best_score",
        "tuner_pick_best_by",
        "best_test_auroc",
        "best_val_auroc",
        "best_n_params",
        "trials",
        "wall_s",
    )
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for s in summaries:
        cells: list[str] = []
        for c in cols:
            v = s.get(c, "")
            if v is None:
                v = ""
            elif isinstance(v, float):
                v = f"{v:.6g}"
            cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
