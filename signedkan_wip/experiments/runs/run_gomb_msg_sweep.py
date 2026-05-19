"""Run ``run_gomb_smoke`` over P-graph MSG / SSG / ABB structures.

Uses the Rust helper ``hymeko_pgraph_dump`` (``cargo build -p hymeko_pgraph
--bin hymeko_pgraph_dump``) to analyse a ``.hymeko`` P-graph, then maps each
selected structure's operating-unit names to Gömb kwargs via
``gomb_pgraph_mapping``.

Modes:

* ``msg`` — print JSON analysis only (no training).
* ``ssg`` — one Gömb smoke per SSG feasible structure (small graphs only;
  ``|O_MSG| <= 30``).
* ``abb`` — single smoke using the cost-minimal ABB unit set.

Example::

    PYTHONPATH=$PWD python -m signedkan_wip.experiments.runs.run_gomb_msg_sweep \\
        --pgraph data/hsikan/sweep_msg_gomb.hymeko \\
        --algorithm ssg \\
        --training data/hsikan/gomb_training.hymeko \\
        --device cpu --max-runs 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from signedkan_wip.src.gomb_pgraph_mapping import (
    merge_structure_knobs,
    smoke_argv_from_knobs,
)
from signedkan_wip.src.hymeko_driver import parse_training


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dump_executable(repo: Path) -> list[str]:
    dbg = repo / "target" / "debug" / "hymeko_pgraph_dump"
    rel = repo / "target" / "release" / "hymeko_pgraph_dump"
    if dbg.is_file():
        return [str(dbg)]
    if rel.is_file():
        return [str(rel)]
    return [
        "cargo",
        "run",
        "-q",
        "-p",
        "hymeko_pgraph",
        "--bin",
        "hymeko_pgraph_dump",
        "--",
    ]


def run_pgraph_dump(repo: Path, pgraph_path: Path, algorithm: str) -> dict[str, Any]:
    prefix = _dump_executable(repo)
    tail = [str(pgraph_path), "--algorithm", algorithm]
    cmd = prefix + tail
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=300,
    )
    if proc.returncode not in (0, 2):
        raise RuntimeError(
            f"hymeko_pgraph_dump failed rc={proc.returncode}\n"
            f"{(proc.stderr or '')[-4000:]}"
        )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("hymeko_pgraph_dump produced empty stdout")
    return json.loads(out)


def _parse_last_smoke_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            return json.loads(line)
    raise ValueError("run_gomb_smoke did not emit trailing JSON summary")


def _run_one_smoke(
    repo: Path,
    *,
    base: dict[str, Any],
    unit_names: list[str],
    dataset: str,
    device: str,
    edge_split: str,
    seed: int,
    timeout_s: int,
) -> dict[str, Any]:
    merged = merge_structure_knobs(unit_names)
    cmd = smoke_argv_from_knobs(
        dataset=dataset,
        device=device,
        edge_split=edge_split,
        seed=seed,
        base=base,
        structure=merged,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_gomb_smoke failed rc={proc.returncode}\n"
            f"{(proc.stderr or '')[-4000:]}"
        )
    row = _parse_last_smoke_json(proc.stdout)
    row["pgraph_units"] = unit_names
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgraph", type=Path, required=True, help="P-graph .hymeko")
    ap.add_argument(
        "--algorithm",
        choices=("msg", "ssg", "abb"),
        default="ssg",
        help="msg = analysis only; ssg = all structures; abb = cost-minimal",
    )
    ap.add_argument(
        "--training",
        type=Path,
        default=None,
        help="Training .hymeko with @gomb_smoke + @load_dataset (optional)",
    )
    ap.add_argument("--dataset", default=None, help="Override dataset name")
    ap.add_argument("--edge-split", default="80_10_10")
    ap.add_argument(
        "--device",
        default=None,
        help="cpu / cuda (default: cuda if available)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--output", default=None, help="Append JSONL per smoke")
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    device = args.device
    if not device:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    repo = _repo_root()
    if args.training is not None:
        base = parse_training(str(args.training))
    else:
        base = parse_training(str(repo / "data" / "hsikan" / "gomb_training.hymeko"))
    base["_python_executable"] = sys.executable

    analysis = run_pgraph_dump(repo, args.pgraph, args.algorithm)
    print(json.dumps({"phase": "pgraph_analysis", **analysis}, indent=2, default=str))

    if not analysis.get("ok"):
        raise SystemExit(2)

    if args.algorithm == "msg":
        return

    dataset = args.dataset or base.get("dataset_name") or "sbm_n200"
    structures: list[list[str]] = []
    if args.algorithm == "abb":
        abb = analysis.get("abb")
        if not abb or not abb.get("units"):
            print("no ABB solution", file=sys.stderr)
            raise SystemExit(3)
        structures = [list(abb["units"])]
    else:
        raw = analysis.get("ssg_structures")
        if not isinstance(raw, list):
            print("missing ssg_structures", file=sys.stderr)
            raise SystemExit(3)
        structures = [list(s) for s in raw if s]

    max_runs = args.max_runs if args.max_runs is not None else len(structures)
    for i, units in enumerate(structures[:max_runs]):
        print(
            f"\n[smoke {i + 1}/{min(len(structures), max_runs)}] units={units}",
            file=sys.stderr,
        )
        row = _run_one_smoke(
            repo,
            base=base,
            unit_names=units,
            dataset=dataset,
            device=device,
            edge_split=args.edge_split,
            seed=args.seed,
            timeout_s=int(args.timeout_s),
        )
        line = json.dumps(row, sort_keys=True, default=str)
        print(line, flush=True)
        if args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
