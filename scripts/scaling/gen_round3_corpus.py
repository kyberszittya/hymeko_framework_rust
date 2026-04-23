#!/usr/bin/env python3
"""Round-3 Exp-4 corpus generator.

Produces four highArity fixtures at |E| ∈ {10, 100, 1000, 10000} with
fixed mean arity d̄ = 3, per `claude_code_round3_experiments_brief.md` §4.1.

Outputs to `hymeko_bench/corpora/03_size_sweep/` with an `index.json`
manifest compatible with `bench_scaling`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_fixtures import gen_high_arity, write_fixture  # type: ignore


REPO = HERE.parents[1]
OUT = REPO / "hymeko_bench" / "corpora" / "03_size_sweep"
META = REPO / "data" / "robotics" / "meta_kinematics.hymeko"

SIZES = [10, 100, 1000, 10_000]
ARITY = 3


def main() -> None:
    if not META.exists():
        sys.exit(f"meta_kinematics not found at {META}")
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for m in SIZES:
        src, stats = gen_high_arity(m=m, d=ARITY, seed=0)
        # Override the default name/family so the manifest is readable.
        stats.family = "03_size_sweep"
        stats.name = f"size_{m:05d}_d{ARITY}"
        # Write the fixture + meta copy alongside.
        manifest.append(
            write_fixture(OUT, "03_size_sweep", stats, src, META)
        )
        print(f"  |E|={m:5d} |V|={stats.n_vertices:5d} d̄={stats.mean_arity:.1f} "
              f"bytes={stats.source_bytes:8d}")

    (OUT / "index.json").write_text(
        json.dumps([asdict(s) for s in manifest], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(manifest)} fixtures to {OUT}")
    print(f"Manifest: {OUT}/index.json")


if __name__ == "__main__":
    main()
