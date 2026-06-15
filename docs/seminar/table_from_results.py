"""Build the signed-link results table for the seminar from ONE source per row.

Honest-protocol tooling (CLAUDE.md): every emitted number is computed from a
single result JSONL — no mixing protocols, no guessing. A dataset with no source
file is written as ``[pending]`` rather than back-filled from a different run.

Each manifest row names: the display dataset, the JSONL it is read from, and the
metric key. The script computes mean ± population-std over the per-seed rows and
emits (a) a Markdown table and (b) a CSV, both into ``figures/``. Baseline numbers
are NOT computed here — under a single-protocol policy they belong in a separate
"reported" block, labelled as literature, never as "our run".

Run:  python docs/seminar/table_from_results.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "signedkan_wip" / "experiments" / "results"
OUT = Path(__file__).resolve().parent / "figures"
METRIC_KEYS = ("test_auroc", "val_auroc", "val_auc_best", "auc")


@dataclass
class Source:
    """One table row, bound to exactly one result file (single-source policy)."""
    dataset: str
    jsonl: str | None            # filename under RESULTS, or None => pending
    label: str = ""              # protocol / run label shown in provenance
    metric_keys: tuple[str, ...] = METRIC_KEYS

    def per_seed(self) -> list[float]:
        if not self.jsonl:
            return []
        path = RESULTS / self.jsonl
        if not path.exists():
            return []
        out: list[float] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in self.metric_keys:
                if row.get(k) is not None:
                    out.append(float(row[k]))
                    break
        return out


@dataclass
class Row:
    dataset: str
    n: int
    mean: float | None
    pstd: float | None
    provenance: str
    per_seed: list[float] = field(default_factory=list)

    @classmethod
    def of(cls, s: Source) -> "Row":
        vals = s.per_seed()
        if not vals:
            return cls(s.dataset, 0, None, None, f"[pending] {s.label or s.jsonl or '—'}")
        mean = st.mean(vals)
        pstd = st.pstdev(vals) if len(vals) > 1 else 0.0
        prov = f"{s.label or s.jsonl} ({len(vals)} seeds)"
        return cls(s.dataset, len(vals), mean, pstd, prov, [round(v, 4) for v in vals])

    @property
    def cell(self) -> str:
        return "[pending]" if self.mean is None else f"{self.mean:.3f} ± {self.pstd:.3f}"


# --- The single authoritative manifest (edit here when a run lands) ------------
# Verified 2026-06-16 against on-disk JSONLs. Bitcoin α/OTC have no local edge_cr
# file (they live on Komondor, not in-repo) => pending. The Gömb-strict cascade
# OOM-crashed on the 7.6 GB local GPU; its row is pending a cloud re-run.
HSIKAN_EDGE_CR = [
    Source("Bitcoin Alpha", None, "edge_cr 5-seed (Komondor — not in repo)"),
    Source("Bitcoin OTC",   None, "edge_cr (no run found)"),
    Source("Slashdot", "slashdot_edge_cr_5seed_2026_05_09.jsonl", "HSiKAN edge_cr 5-seed"),
    Source("Epinions", "epinions_edge_cr_5seed_2026_05_09.jsonl", "HSiKAN edge_cr 5-seed"),
]
GOMB_STRICT = [
    Source(d, None, "Gömb-strict (OOM on local GPU — re-run on Komondor/GCP)")
    for d in ("Bitcoin Alpha", "Bitcoin OTC", "Slashdot", "Epinions")
]


def build(name: str, sources: list[Source]) -> list[Row]:
    return [Row.of(s) for s in sources]


def write_markdown(path: Path, title: str, rows: list[Row]) -> None:
    lines = [f"### {title}", "",
             "| Dataset | AUROC mean ± pstd | n seeds | source |",
             "|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r.dataset} | {r.cell} | {r.n or '—'} | {r.provenance} |")
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "mean", "pstd", "n_seeds", "provenance", "per_seed"])
        for r in rows:
            w.writerow([r.dataset, r.mean if r.mean is not None else "",
                        r.pstd if r.pstd is not None else "", r.n, r.provenance,
                        " ".join(map(str, r.per_seed))])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, srcs, slug in [
        ("HSiKAN — edge_cr 5-seed paired (verified source)", HSIKAN_EDGE_CR, "hsikan_edge_cr"),
        ("Gömb-strict cascade", GOMB_STRICT, "gomb_strict"),
    ]:
        rows = build(name, srcs)
        write_markdown(OUT / f"results_table_{slug}.md", name, rows)
        write_csv(OUT / f"results_table_{slug}.csv", rows)
        print(f"\n{name}")
        for r in rows:
            print(f"  {r.dataset:14s} {r.cell:18s} {r.provenance}")
    print(f"\nwrote results_table_*.{{md,csv}} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
