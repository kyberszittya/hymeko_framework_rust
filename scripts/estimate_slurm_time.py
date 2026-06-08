#!/usr/bin/env python3
"""Estimate a SLURM ``--time`` budget for HSiKAN audit cells.

The estimator measures n_edges from the raw dataset file at call
time, then interpolates the expected per-cell wall from the anchors
in ``docs/komondor_setup/wall_calibration.yaml``. No per-dataset
hardcoded budgets; adding a new dataset is zero-config.

Input
-----
Either:
  --cells DATASET:MODE:CACHE [DATASET:MODE:CACHE ...]
    explicit cells, e.g.  ``slashdot:shuffle:cold epinions:real:cold``
  --class {tiny,medium,long}
    canonical HSiKAN-edge_cr cell classes.
  --measure DATASET
    just print the measured n_edges for the dataset and exit.

Output
------
A single SLURM HH:MM:SS string on stdout (e.g. ``00:15:00``).

Side effects
------------
None. Diagnostics on stderr; exits non-zero with a useful message if
the chosen budget would yield TimeEff below the YAML's
``min_time_efficiency`` floor.

See:
  docs/komondor_setup/wall_calibration.yaml      -- anchors + model
  reports/2026-06-04-kifu-resource-eff-response.md
  CLAUDE.md §6.5 #16                              -- HPC sizing rule
"""
from __future__ import annotations

import argparse
import bisect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:
    print("error: pyyaml not installed; pip install pyyaml", file=sys.stderr)
    sys.exit(2)


REPO = Path(__file__).resolve().parents[1]
CALIB_PATH = REPO / "docs" / "komondor_setup" / "wall_calibration.yaml"
RAW_DATA_DIR = REPO / "signedkan_wip" / "data"

# Map dataset slug -> raw file in RAW_DATA_DIR.
RAW_FILENAME = {
    "bitcoin_alpha": "bitcoin_alpha.csv",
    "bitcoin_otc":   "bitcoin_otc.csv",
    "slashdot":      "slashdot.txt",
    "epinions":      "epinions.txt",
}


# Canonical HSiKAN-edge_cr cell classes (matches the orchestrator).
CLASS_CELLS: dict[str, list[tuple[str, str, str]]] = {
    "tiny": [
        ("bitcoin_alpha", "real",    "cold"),
        ("bitcoin_alpha", "shuffle", "cold"),
        ("bitcoin_otc",   "real",    "cold"),
        ("bitcoin_otc",   "shuffle", "cold"),
        ("slashdot",      "real",    "warm"),
    ],
    "medium": [
        ("slashdot",      "shuffle", "cold"),
    ],
    "long": [
        ("epinions",      "real",    "cold"),
        ("epinions",      "shuffle", "cold"),
    ],
}


@dataclass(frozen=True)
class Anchor:
    dataset: str
    n_edges: int
    n_nodes: int
    wall_cold: int
    wall_warm: int


@dataclass(frozen=True)
class Calib:
    anchors: list[Anchor]                    # sorted by n_edges asc
    warm_cache_constant_s: int
    safety_multiplier: float
    min_wall_s: int
    max_wall_s: int
    min_time_efficiency: float

    @classmethod
    def load(cls, path: Path) -> "Calib":
        if not path.exists():
            raise SystemExit(f"calibration YAML not found: {path}")
        with path.open() as f:
            y = yaml.safe_load(f)
        anchors = sorted(
            (Anchor(
                dataset=a["dataset"],
                n_edges=int(a["n_edges"]),
                n_nodes=int(a["n_nodes"]),
                wall_cold=int(a["wall_cold"]),
                wall_warm=int(a["wall_warm"]),
            ) for a in y["anchors"]),
            key=lambda a: a.n_edges,
        )
        ex = y["extrapolation"]
        return cls(
            anchors=anchors,
            warm_cache_constant_s=int(y["warm_cache_constant_s"]),
            safety_multiplier=float(y["safety_multiplier"]),
            min_wall_s=int(ex["min_wall_s"]),
            max_wall_s=int(ex["max_wall_s"]),
            min_time_efficiency=float(y["min_time_efficiency"]),
        )


def count_edges(dataset: str, raw_dir: Path = RAW_DATA_DIR) -> int:
    """Count non-comment non-blank lines in the raw dataset file."""
    fname = RAW_FILENAME.get(dataset)
    if not fname:
        raise SystemExit(f"unknown dataset (no raw filename mapping): {dataset}")
    fpath = raw_dir / fname
    if not fpath.exists():
        raise SystemExit(f"raw dataset file not found: {fpath}")
    n = 0
    with fpath.open() as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def loglog_interpolate_cold(calib: Calib, n_edges: int) -> int:
    """Piecewise log-log linear interpolation of cold-cache wall.

    For n_edges outside the anchor range:
      below smallest -> smallest anchor's wall_cold (no faster than
                        the training+init floor).
      above largest  -> extrapolate using the slope of the last two
                        anchors, capped at max_wall_s.
    """
    if n_edges <= 0:
        raise SystemExit(f"non-positive n_edges: {n_edges}")
    xs = [a.n_edges for a in calib.anchors]
    ys = [a.wall_cold for a in calib.anchors]

    # Below smallest anchor -> use its wall.
    if n_edges <= xs[0]:
        return max(calib.min_wall_s, ys[0])

    # Above largest anchor -> extrapolate from last two.
    if n_edges >= xs[-1]:
        # log-log slope from last two anchors.
        lx_a, lx_b = math.log10(xs[-2]), math.log10(xs[-1])
        ly_a, ly_b = math.log10(ys[-2]), math.log10(ys[-1])
        slope = (ly_b - ly_a) / (lx_b - lx_a)
        intercept = ly_b - slope * lx_b
        log_y = slope * math.log10(n_edges) + intercept
        wall = int(math.ceil(10 ** log_y))
        return min(calib.max_wall_s, max(calib.min_wall_s, wall))

    # Bracketed: locate adjacent anchors and interpolate in log-log.
    i = bisect.bisect_right(xs, n_edges)
    lx_a, lx_b = math.log10(xs[i - 1]), math.log10(xs[i])
    ly_a, ly_b = math.log10(ys[i - 1]), math.log10(ys[i])
    t = (math.log10(n_edges) - lx_a) / (lx_b - lx_a)
    log_y = ly_a + t * (ly_b - ly_a)
    wall = int(math.ceil(10 ** log_y))
    return min(calib.max_wall_s, max(calib.min_wall_s, wall))


def estimate_one(calib: Calib, dataset: str, mode: str, cache: str,
                 raw_dir: Path = RAW_DATA_DIR) -> dict:
    n_edges = count_edges(dataset, raw_dir=raw_dir)
    if cache == "warm":
        wall = calib.warm_cache_constant_s
        method = "warm-constant"
    elif cache == "cold":
        wall = loglog_interpolate_cold(calib, n_edges)
        method = "cold-loglog-interp"
    else:
        raise SystemExit(f"bad cache state '{cache}'; use cold|warm")
    return {
        "dataset": dataset, "mode": mode, "cache": cache,
        "n_edges": n_edges, "wall_s": wall, "method": method,
    }


def estimate_class(calib: Calib, cells: Iterable[tuple[str, str, str]],
                   raw_dir: Path = RAW_DATA_DIR
                   ) -> tuple[int, list[dict]]:
    info = [estimate_one(calib, ds, mode, cache, raw_dir=raw_dir)
            for ds, mode, cache in cells]
    worst = max(c["wall_s"] for c in info)
    with_safety = int(math.ceil(worst * calib.safety_multiplier))
    return with_safety, info


def fmt_slurm_time(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_cell_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(
            f"bad cell spec '{spec}'; expected DATASET:MODE:CACHE")
    return parts[0], parts[1], parts[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--cells", nargs="+",
                     help="explicit cells: DATASET:MODE:CACHE ...")
    sel.add_argument("--class", dest="cls",
                     choices=sorted(CLASS_CELLS.keys()),
                     help="canonical class (tiny/medium/long)")
    sel.add_argument("--measure", metavar="DATASET",
                     help="just print measured n_edges for DATASET and exit")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR,
                   help="directory holding raw dataset files; "
                        "default %(default)s")
    p.add_argument("--explain", action="store_true",
                   help="print per-cell breakdown to stderr")
    p.add_argument("--no-eff-check", action="store_true",
                   help="skip the min_time_efficiency floor check")
    args = p.parse_args(argv)

    if args.measure:
        print(count_edges(args.measure, raw_dir=args.raw_dir))
        return 0

    calib = Calib.load(CALIB_PATH)
    cells = CLASS_CELLS[args.cls] if args.cls \
            else [parse_cell_spec(s) for s in args.cells]

    budget_s, info = estimate_class(calib, cells, raw_dir=args.raw_dir)
    slurm = fmt_slurm_time(budget_s)

    if not args.no_eff_check and info:
        fastest = min(c["wall_s"] for c in info)
        eff = fastest / budget_s if budget_s > 0 else 0.0
        if eff < calib.min_time_efficiency:
            print(
                f"error: budget {slurm} ({budget_s}s) yields TimeEff "
                f"{eff:.1%} on fastest cell ({fastest}s); below floor "
                f"{calib.min_time_efficiency:.0%}. Split the class.",
                file=sys.stderr,
            )
            return 3

    if args.explain:
        print(f"cells (n={len(info)}):", file=sys.stderr)
        for c in info:
            print(
                f"  {c['dataset']:<14} {c['mode']:<8} {c['cache']:<5} "
                f"n_edges={c['n_edges']:>8d}  wall={c['wall_s']:>6}s  "
                f"({c['method']})",
                file=sys.stderr,
            )
        worst = max(c["wall_s"] for c in info)
        fastest = min(c["wall_s"] for c in info)
        print(
            f"worst={worst}s × safety={calib.safety_multiplier} = "
            f"{budget_s}s = {slurm}; "
            f"TimeEff on fastest ({fastest}s) = "
            f"{fastest/budget_s:.1%}",
            file=sys.stderr,
        )

    print(slurm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
