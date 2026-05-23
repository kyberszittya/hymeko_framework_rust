#!/usr/bin/env python3
"""
bench_competitors.py — subprocess-based timing of the standard single-target
conversion stack (xacro, gz sdf, mujoco) for head-to-head comparison
against HyMeKo's in-process multi-format emission.

Stages timed:
  xacro_urdf : `xacro <in.urdf>` (URDF through xacro — baseline URDF cost)
  gz_sdf     : `gz sdf -p <in.urdf>` (URDF → SDF)
  mujoco_mjcf: `python3 urdf_to_mjcf.py <in.urdf>` (URDF → MJCF)
  bundle     : all three run sequentially (coherent-bundle cost)

Each stage pays its own subprocess startup — that is the point of the
comparison. HyMeKo's numbers come from hymeko_bench and include no
subprocess startup, because emission is in-process by design.

Writes one CSV row per (fixture × rep × stage) with schema compatible
with scaling_results.csv, plus a `tool` column so a merged CSV
distinguishes HyMeKo from the competitor stack.

Usage:
    python bench_competitors.py --urdf-fixtures ./urdf_fixtures \
                                --out competitor_results.csv --reps 10
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

# URDF→MJCF conversion runs in a subprocess so startup cost is counted.
URDF_TO_MJCF_SCRIPT = r"""
import sys, mujoco
spec = mujoco.MjSpec.from_file(sys.argv[1])
sys.stdout.write(spec.to_xml())
"""


def write_helper_script(path: Path) -> None:
    path.write_text(URDF_TO_MJCF_SCRIPT, encoding="utf-8")


def run_timed(cmd: list[str]) -> tuple[int, float]:
    """Return (output_bytes, wall_ns)."""
    t0 = time.monotonic_ns()
    result = subprocess.run(
        cmd, capture_output=True, check=False,
    )
    dt = time.monotonic_ns() - t0
    if result.returncode != 0:
        return (-1, dt)
    return (len(result.stdout), dt)


def run_bundle(xacro_cmd: list[str], gz_cmd: list[str],
               mjcf_cmd: list[str]) -> tuple[int, float]:
    """Wall-clock for xacro + gz + mujoco in sequence."""
    t0 = time.monotonic_ns()
    for cmd in (xacro_cmd, gz_cmd, mjcf_cmd):
        r = subprocess.run(cmd, capture_output=True, check=False)
        if r.returncode != 0:
            return (-1, time.monotonic_ns() - t0)
    dt = time.monotonic_ns() - t0
    return (0, dt)


def bench_fixture(writer: csv.DictWriter, fixtures_root: Path,
                  entry: dict, reps: int, warmup: int,
                  helper: Path) -> None:
    urdf_path = fixtures_root / entry["path"]

    xacro_cmd = ["xacro", str(urdf_path)]
    gz_cmd = ["gz", "sdf", "-p", str(urdf_path)]
    mjcf_cmd = ["python3", str(helper), str(urdf_path)]

    for _ in range(warmup):
        run_timed(xacro_cmd)
        run_timed(gz_cmd)
        run_timed(mjcf_cmd)

    for rep in range(reps):
        for stage, cmd in (
            ("xacro_urdf", xacro_cmd),
            ("gz_sdf",     gz_cmd),
            ("mujoco_mjcf", mjcf_cmd),
        ):
            out_bytes, dt = run_timed(cmd)
            writer.writerow({
                "tool": "competitor",
                "family": entry["family"], "name": entry["name"],
                "n_vertices": entry["n_vertices"],
                "n_hyperedges": entry["n_hyperedges"],
                "mean_arity": entry["mean_arity"],
                "source_bytes": entry["source_bytes"],
                "rep": rep, "stage": stage,
                "wall_ns": dt,
                "output_bytes": out_bytes if out_bytes >= 0 else 0,
            })

        # coherent 3-format bundle
        out_bytes, dt = run_bundle(xacro_cmd, gz_cmd, mjcf_cmd)
        writer.writerow({
            "tool": "competitor",
            "family": entry["family"], "name": entry["name"],
            "n_vertices": entry["n_vertices"],
            "n_hyperedges": entry["n_hyperedges"],
            "mean_arity": entry["mean_arity"],
            "source_bytes": entry["source_bytes"],
            "rep": rep, "stage": "bundle_3fmt",
            "wall_ns": dt, "output_bytes": 0,
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf-fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10,
                    help="Repetitions per fixture (default 10 — subprocess"
                         " startup dominates, so fewer reps are fine).")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=None,
                    help="Skip fixtures with n_vertices > this")
    ap.add_argument("--family", type=str, default=None)
    args = ap.parse_args()

    manifest = json.loads(
        (args.urdf_fixtures / "index.json").read_text(encoding="utf-8"))
    manifest = [e for e in manifest
                if (args.family is None or e["family"] == args.family)
                and (args.max_size is None or e["n_vertices"] <= args.max_size)]

    helper = args.out.parent / "_urdf_to_mjcf_helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    write_helper_script(helper)

    fieldnames = ["tool", "family", "name", "n_vertices", "n_hyperedges",
                  "mean_arity", "source_bytes", "rep", "stage", "wall_ns",
                  "output_bytes"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, entry in enumerate(manifest):
            print(f"[{i+1}/{len(manifest)}] {entry['name']} "
                  f"({entry['n_vertices']} V, {entry['n_hyperedges']} E)",
                  flush=True)
            bench_fixture(w, args.urdf_fixtures, entry,
                          args.reps, args.warmup, helper)
            f.flush()
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
