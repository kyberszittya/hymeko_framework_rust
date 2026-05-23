"""HSIKAN architecture-search driver — Phase 7 (2026-05-19).

Parses an HSIKAN architecture-search P-graph (default
``data/hsikan/sweep_msg.hymeko``) via the Rust ``hymeko_pgraph_dump``
binary, lifts the ABB-selected operating units onto HSIKAN
``run_compare.run_one`` kwargs through
:mod:`signedkan_wip.src.hsikan_pgraph_mapping`, and (optionally)
launches one HSIKAN training run per ABB / SSG selection.

Use::

    PYTHONPATH=$PWD python -m signedkan_wip.experiments.runs.run_hsikan_msg_sweep \\
        --pgraph data/hsikan/sweep_msg.hymeko \\
        --algorithm abb

By default the driver does **not** start training: it just emits the
P-graph analysis + the mapped HSIKAN kwargs + the canonical / extension
Friedler certificate. Pass ``--train`` to actually invoke
``run_compare.run_one`` for each picked architecture.

Parallel in shape to ``run_gomb_msg_sweep.py``; the HSIKAN side
re-uses the same Rust binary and the same ``hymeko_pgraph_dump``
JSON DTO (including the new ``canonical_*`` / ``extension_*`` fields
the Phase 7 PR landed).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PGRAPH = REPO_ROOT / "data" / "hsikan" / "sweep_msg.hymeko"


def _find_dump_binary(repo: Path) -> Path:
    dbg = repo / "target" / "debug" / "hymeko_pgraph_dump"
    rel = repo / "target" / "release" / "hymeko_pgraph_dump"
    if rel.exists():
        return rel
    if dbg.exists():
        return dbg
    # Build it if absent — assumes cargo is on PATH.
    subprocess.run(
        ["cargo", "build", "-p", "hymeko_pgraph", "--bin", "hymeko_pgraph_dump"],
        cwd=str(repo),
        check=True,
    )
    if rel.exists():
        return rel
    return dbg


def run_pgraph_dump(
    repo: Path,
    pgraph_path: Path,
    algorithm: str,
    relaxed_msg: bool,
    weights: str | None = None,
) -> dict[str, Any]:
    bin_path = _find_dump_binary(repo)
    cmd = [str(bin_path), str(pgraph_path), "--algorithm", algorithm]
    if relaxed_msg:
        cmd.append("--relaxed-msg")
    if weights:
        cmd.extend(["--weights", weights])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo))
    if proc.returncode != 0:
        raise RuntimeError(
            f"hymeko_pgraph_dump failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if not proc.stdout.strip():
        raise RuntimeError("hymeko_pgraph_dump produced empty stdout")
    return json.loads(proc.stdout)


def _format_certificate(cert: dict[str, Any] | None, label: str) -> str:
    if cert is None:
        return f"  {label}: n/a"
    status = cert.get("status", "?")
    if status == "PASS":
        return f"  {label}: PASS"
    tags = ",".join(cert.get("violation_tags", []))
    offenders_str = "; ".join(
        f"{tag}=[{','.join(names)}]"
        for tag, names in cert.get("offenders", [])
    )
    return f"  {label}: FAIL [{tags}] — {offenders_str}"


def _print_analysis_summary(analysis: dict[str, Any]) -> None:
    print(f"description: {analysis.get('description')}")
    print(f"algorithm:   {analysis.get('algorithm')}")
    print(f"strict_no_excess: {analysis.get('strict_no_excess', True)}")
    print(_format_certificate(analysis.get("canonical_full"), "canonical (full schema)"))
    print(_format_certificate(analysis.get("extension_full"), "extension (full schema)"))
    print(_format_certificate(analysis.get("canonical_abb_subschema"), "canonical (ABB sub-schema)"))
    print(_format_certificate(analysis.get("extension_abb_subschema"), "extension (ABB sub-schema)"))
    print(f"msg_units ({len(analysis.get('msg_units', []))}): {analysis.get('msg_units')}")
    abb = analysis.get("abb")
    if abb:
        print(f"abb units ({len(abb.get('units', []))}): {abb.get('units')} cost={abb.get('cost')}")


def _selected_unit_sets(analysis: dict[str, Any], algorithm: str) -> list[list[str]]:
    """Return the list of unit-name sets to evaluate.

    For ``abb``: a single list (the cost-minimal selection).
    For ``ssg``: every feasible structure.
    For ``msg``: empty (no concrete selection; analysis-only).
    """
    if algorithm == "abb":
        abb = analysis.get("abb")
        if not abb or not abb.get("units"):
            return []
        return [list(abb["units"])]
    if algorithm == "ssg":
        return list(analysis.get("ssg_structures") or [])
    return []


def _map_one(unit_names: list[str], dataset: str, seed: int, base: dict[str, Any]) -> dict[str, Any]:
    from signedkan_wip.src.hsikan_pgraph_mapping import (
        merge_structure_knobs,
        run_one_kwargs,
    )

    merged = merge_structure_knobs(unit_names)
    kwargs = run_one_kwargs(
        dataset=dataset, seed=seed, structure=merged, base=base,
    )
    return {
        "pgraph_units": unit_names,
        "merged_structure": merged,
        "run_one_kwargs": kwargs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgraph", type=Path, default=DEFAULT_PGRAPH,
                    help="P-graph .hymeko (default: data/hsikan/sweep_msg.hymeko)")
    ap.add_argument("--algorithm", choices=("msg", "ssg", "abb"), default="abb")
    ap.add_argument("--relaxed-msg", action="store_true",
                    help="Forward to hymeko_pgraph_dump --relaxed-msg")
    ap.add_argument("--weights", type=str, default=None,
                    help="Phase 10: forward to hymeko_pgraph_dump --weights "
                         "as a comma-separated list aligned with the "
                         "alphabetised cost_dimensions of the fixture")
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--output", type=Path, default=None,
                    help="Append JSONL per (selection, seed)")
    ap.add_argument("--train", action="store_true",
                    help="Invoke run_compare.run_one for each selection (default: dry-run)")
    args = ap.parse_args()

    analysis = run_pgraph_dump(
        REPO_ROOT, args.pgraph, args.algorithm, args.relaxed_msg,
        weights=args.weights,
    )
    _print_analysis_summary(analysis)

    if not analysis.get("ok"):
        print("analysis FAILED:", analysis.get("parse_error") or analysis.get("lower_error"),
              file=sys.stderr)
        sys.exit(2)

    selections = _selected_unit_sets(analysis, args.algorithm)
    if not selections:
        print(f"\nno concrete selection to map for algorithm={args.algorithm}")
        return

    base: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for sel in selections:
        for seed in args.seeds:
            row = _map_one(sel, args.dataset, seed, base)
            row["dataset"] = args.dataset
            row["seed"] = seed
            row["strict_no_excess"] = analysis.get("strict_no_excess", True)
            row["canonical_abb_subschema_status"] = (
                analysis.get("canonical_abb_subschema", {}).get("status")
                if analysis.get("canonical_abb_subschema") else None
            )
            row["extension_abb_subschema_status"] = (
                analysis.get("extension_abb_subschema", {}).get("status")
                if analysis.get("extension_abb_subschema") else None
            )
            if args.train:
                try:
                    from signedkan_wip.experiments.runs.run_compare import run_one
                    # Phase 8 (2026-05-19): run_one now accepts m_cycles
                    # directly — no env-var workaround needed.
                    kw = dict(row["run_one_kwargs"])
                    print(f"\n[training] units={sel} seed={seed} kwargs={kw}")
                    result = run_one(**kw)
                    row["training_result"] = result
                except Exception as exc:  # noqa: BLE001 — surface the error
                    row["training_error"] = repr(exc)
            rows.append(row)
            print(f"\nselection {sel} seed={seed}:")
            print(f"  merged structure  = {row['merged_structure']}")
            print(f"  run_one_kwargs    = {row['run_one_kwargs']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"\nwrote {len(rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
