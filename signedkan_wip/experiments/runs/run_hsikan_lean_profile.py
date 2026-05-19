"""HSiKAN **enumerator-first lean sweep** — small ``hidden_dim`` × k-pool recipe.

Runs ``run_final_cell.cell_signed_graph`` in **isolated subprocesses** with a
**scrubbed** ``HSIKAN_*`` environment so parent-shell leakage cannot silently
change the enumerator path.

**Design goal (research line):** push HSiKAN-family models **down in
``|V|·hidden`` + spline overhead** while preserving signal by spending budget
on **better k-structures** instead of width:

* **k-enumeration** — ``HSIKAN_TOPK_MODE`` / ``HSIKAN_TOPK_K`` (per-vertex
  top-m, global caps) control how many tuple columns enter the sparse
  incidence head.
* **ABB** — ``HSIKAN_USE_PER_VERTEX_ABB`` + ``HSIKAN_PER_VERTEX_ABB_*``
  (global-min fullness gate) trims redundant / low-value cycles so the
  same ``hidden`` sees less noisy incidence.
* **SSG (Selective Structural Gating)** — here mapped to **vertex
  pre-filter** ``HSIKAN_VERTEX_FILTER`` / ``HSIKAN_VERTEX_FILTER_MIN_DEGREE``
  (see ``runtime_config.TopKConfig``): enumerate / score only on a
  degree-selected **subgraph shell** before tuple materialisation.

This does **not** change core ``MixedAritySignedKAN`` maths; it is a
**reproducible harness** for comparing *width vs enumerator* trade-offs.

From the repo root, after ``uv sync --group ml --all-packages`` (see
``README.md`` — Python (uv))::

    export PYTHONPATH=.
    PY="$(uv run python -c "import sys, torch; print(sys.executable)")"
    uv run python -m signedkan_wip.experiments.runs.run_hsikan_lean_profile \\
        --python "${PY}" \\
        --datasets bitcoin_alpha bitcoin_otc \\
        --seeds 0 1 2 --hidden 8 12 16 \\
        --profiles clean_baseline pv_k128_abb_g10 pv_k64_abb_ssg_deg3 \\
        --n-epochs 80 \\
        --out reports/hsikan_lean_profile.jsonl

With any torch-capable ``python``, pass ``--python`` explicitly (required under
minimal ``PATH``, e.g. ``systemd-run``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Named profiles: values are **complete** HSIKAN_* overrides for that recipe
# (after ``build_child_env`` strips every pre-existing ``HSIKAN_*`` key).
PROFILE_ENV: dict[str, dict[str, str]] = {
    "clean_baseline": {
        # Explicitly disable top-K / ABB / vertex filter → legacy fall-through
        # where ``HSIKAN_TOPK_MODE`` is empty (see ``runtime_config.TopKConfig``).
        "HSIKAN_TOPK_MODE": "",
        "HSIKAN_TOPK_K": "16",
        "HSIKAN_USE_PER_VERTEX_ABB": "0",
        "HSIKAN_VERTEX_FILTER": "none",
    },
    "pv_k128": {
        "HSIKAN_TOPK_MODE": "per_vertex",
        "HSIKAN_TOPK_K": "128",
        "HSIKAN_USE_PER_VERTEX_ABB": "0",
        "HSIKAN_VERTEX_FILTER": "none",
    },
    "pv_k128_abb_g10": {
        "HSIKAN_TOPK_MODE": "per_vertex",
        "HSIKAN_TOPK_K": "128",
        "HSIKAN_USE_PER_VERTEX_ABB": "1",
        "HSIKAN_USE_PER_VERTEX_ABB_MODE": "global",
        "HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE": "1.0",
        "HSIKAN_VERTEX_FILTER": "none",
    },
    "pv_k64_abb_g10": {
        "HSIKAN_TOPK_MODE": "per_vertex",
        "HSIKAN_TOPK_K": "64",
        "HSIKAN_USE_PER_VERTEX_ABB": "1",
        "HSIKAN_USE_PER_VERTEX_ABB_MODE": "global",
        "HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE": "1.0",
        "HSIKAN_VERTEX_FILTER": "none",
    },
    "pv_k64_abb_ssg_deg3": {
        "HSIKAN_TOPK_MODE": "per_vertex",
        "HSIKAN_TOPK_K": "64",
        "HSIKAN_USE_PER_VERTEX_ABB": "1",
        "HSIKAN_USE_PER_VERTEX_ABB_MODE": "global",
        "HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE": "1.0",
        "HSIKAN_VERTEX_FILTER": "degree",
        "HSIKAN_VERTEX_FILTER_MIN_DEGREE": "3",
    },
}


def build_child_env(profile_vars: dict[str, str]) -> dict[str, str]:
    """Return ``os.environ`` for a child: strip ``HSIKAN_*``, apply ``profile_vars``.

    Preserves ``HSIKAN_TORCH_COMPILE`` from the parent when set (compile is
    orthogonal to enumerator choice).
    """
    tc = os.environ.get("HSIKAN_TORCH_COMPILE")
    child = {k: v for k, v in os.environ.items() if not k.startswith("HSIKAN_")}
    child.update(profile_vars)
    if tc is not None:
        child["HSIKAN_TORCH_COMPILE"] = tc
    return child


def _parse_last_cell_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _run_cell_subprocess(
    *,
    py: str,
    dataset: str,
    seed: int,
    hidden: int,
    n_epochs: int,
    max_k4: int,
    profile_vars: dict[str, str],
    timeout_s: int,
    device: str,
) -> dict[str, Any]:
    cmd = [
        py, "-m", "signedkan_wip.experiments.runs.run_final_cell",
        "--dataset", dataset,
        "--model", "HSiKAN",
        "--hidden", str(hidden),
        "--n-epochs", str(n_epochs),
        "--max-k4", str(max_k4),
        "--seed", str(seed),
    ]
    env = build_child_env(profile_vars)
    if device != "auto":
        env["HSIKAN_DEVICE"] = device
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    row = _parse_last_cell_json(proc.stdout) or {}
    row["returncode"] = int(proc.returncode)
    row["profile_env"] = profile_vars
    if proc.returncode != 0:
        row["stderr_tail"] = (proc.stderr or "")[-6000:]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument(
        "--hidden", nargs="+", type=int, default=[8, 12, 16],
        help="One or more hidden widths for MixedAritySignedKAN base.",
    )
    ap.add_argument(
        "--profiles", nargs="+", choices=sorted(PROFILE_ENV), required=True,
    )
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--max-k4", type=int, default=200_000)
    ap.add_argument(
        "--timeout-s", type=int, default=7200,
        help="Per-cell subprocess wall limit.",
    )
    ap.add_argument("--out", type=Path, required=True, help="JSONL append path")
    ap.add_argument(
        "--python",
        default=None,
        metavar="EXE",
        help=(
            "Python executable for each ``run_final_cell`` subprocess "
            "(default: ``sys.executable``). Under ``systemd-run`` or cron, "
            "PATH often points at a system ``python3`` without torch — set this "
            "to your conda/venv interpreter."
        ),
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Training device for each subprocess (sets ``HSIKAN_DEVICE``). "
            "Use ``cpu`` when the GPU is shared so sweeps do not OOM mid-grid."
        ),
    )
    args = ap.parse_args()

    py_raw = args.python or sys.executable
    # Keep venv wrapper paths (e.g. .venv/bin/python3); do not resolve
    # symlinks to the base interpreter, otherwise site-packages context is lost.
    py = str(Path(py_raw).expanduser())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for profile in args.profiles:
            penv = PROFILE_ENV[profile]
            for h in args.hidden:
                for seed in args.seeds:
                    t0 = time.perf_counter()
                    row = _run_cell_subprocess(
                        py=py,
                        dataset=ds,
                        seed=seed,
                        hidden=h,
                        n_epochs=args.n_epochs,
                        max_k4=args.max_k4,
                        profile_vars=penv,
                        timeout_s=args.timeout_s,
                        device=args.device,
                    )
                    row["lean_dataset"] = ds
                    row["lean_profile"] = profile
                    row["lean_hidden"] = h
                    row["lean_seed"] = seed
                    row["lean_wall_s"] = time.perf_counter() - t0
                    row["lean_python"] = py
                    row["lean_device"] = args.device
                    with args.out.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                    auc = row.get("auc")
                    npar = row.get("n_params")
                    print(
                        f"[lean] {ds} profile={profile} h={h} seed={seed} "
                        f"dev={args.device} auc={auc} n_params={npar} "
                        f"rc={row['returncode']} wall={row['lean_wall_s']:.1f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
