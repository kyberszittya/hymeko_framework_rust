"""HSiKAN-mixed × protocol-axis smoke under the Composite P-graph regime.

Ties the canonical+no-excess (Composite) P-graph solve to a real
HSiKAN-mixed run:

    sweep_msg_mixed_protocols.hymeko
      → hymeko_pgraph_dump --algorithm abb --regime canonical+no-excess
      → abb.units
      → hsikan_pgraph_mapping.merge_structure_knobs
      → HSIKAN_* env + cell_signed_graph CLI args
      → bitcoin_alpha abbreviated run
      → assert peak-RSS / wall budget, emit a provenance row.

The structural primitive is FIXED to the mixed cycles+walks family
(``c3,c4,w2,w3``); the P-graph ranges over the *other protocols*
(attention / edge-gate / direct-messaging / hidden / training-length).
See ``docs/plans/2026-05-27-hsikan-mixed-composite-regime/``.

Usage
-----
    PYTHONPATH=$PWD systemd-run --user --scope -p MemoryMax=16G \\
        python -m hymeko_neuro.experiments.runs.run_hsikan_mixed_composite_smoke \\
            --regime canonical+no-excess --dataset bitcoin_alpha \\
            --seed 0 --n-epochs 3 --results-file /tmp/composite_smoke.jsonl

``--dry-run`` stops after the solve+map and prints the selected units,
the merged knob dict, and the env/CLI it *would* launch — no torch, no
GPU. The unit/integration tests drive the dry-run path.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hymeko_neuro.experiments.hsikan_pgraph_mapping import merge_structure_knobs

# NOTE: these two path helpers mirror `run_gomb_msg_sweep._repo_root` /
# `._dump_executable`. They are inlined rather than imported because that
# sibling module pulls in the torch + `hymeko` PyO3 wheel chain at import
# time (via `hymeko_driver`), which would make this driver unimportable
# in any environment lacking the wheel — including the torch-free unit
# tests and the dry-run path. Path probing is not the duplicated
# *algorithm* CLAUDE.md §6.1 targets; the heavy-import coupling is the
# larger evil here.


def _repo_root() -> Path:
    # parents[3] = repo root (workspace Cargo.toml, target/, data/hsikan/).
    return Path(__file__).resolve().parents[3]


def _dump_executable(repo: Path) -> list[str]:
    for build in ("debug", "release"):
        cand = repo / "target" / build / "hymeko_pgraph_dump"
        if cand.is_file():
            return [str(cand)]
    return ["cargo", "run", "-q", "-p", "hymeko_pgraph", "--bin",
            "hymeko_pgraph_dump", "--"]

DEFAULT_SWEEP = Path("data/hsikan/sweep_msg_mixed_protocols.hymeko")
DEFAULT_REGIME = "canonical+no-excess"
RUN_CELL_MODULE = "hymeko_neuro.experiments.runs.run_final_cell"

# Contract preservation (CLAUDE.md §3 / plan risk note): enabling the
# attention head disables cycle-batching (run_final_cell.py:~504), so
# peak GPU memory scales with the FULL per-edge cycle/walk set and OOMs
# a consumer GPU even on Bitcoin Alpha. The attention branch must
# therefore inherit an enumeration cap. Measured 2026-05-27 on a
# 7.6 GiB RTX 2070S: uncapped attn=dot OOM'd; per-vertex top-K=8 fit
# (1.x GiB, AUC 0.917). These defaults are applied automatically when
# an ABB selection turns attention on (callers may pre-set either env
# var to override).
_ATTENTION_KINDS = ("dot", "quaternion")
ATTENTION_TOPK_MODE = "per_vertex"
ATTENTION_TOPK_K = 8


def solve_regime(
    repo: Path, sweep: Path, regime: str, *, algorithm: str = "abb"
) -> dict[str, Any]:
    """Run ``hymeko_pgraph_dump`` under ``regime`` and return parsed JSON.

    Preconditions: ``sweep`` exists; the dump binary is buildable.
    Postconditions: returns a dict with ``parse_error``/``lower_error``
    both ``None`` (raises otherwise) and a populated ``abb``/``msg_units``.
    """
    cmd = _dump_executable(repo) + [
        str(sweep),
        "--algorithm",
        algorithm,
        "--regime",
        regime,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(repo), check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hymeko_pgraph_dump failed rc={proc.returncode}\n{proc.stderr}"
        )
    data = json.loads(proc.stdout)
    if data.get("parse_error") or data.get("lower_error"):
        raise RuntimeError(
            f"pgraph solve error: parse={data.get('parse_error')!r} "
            f"lower={data.get('lower_error')!r}"
        )
    return data


def structure_to_env(
    structure: dict[str, Any], attention_topk_k: int = ATTENTION_TOPK_K
) -> tuple[dict[str, str], dict[str, Any]]:
    """Translate a merged knob dict into (HSIKAN_* env patch, CLI kwargs).

    Boolean/str protocol knobs become env vars consumed by
    ``RuntimeConfig`` inside ``cell_signed_graph``; ``hidden`` and
    ``n_epochs`` become ``run_final_cell`` CLI args. The mixed-tuples
    family is forwarded unconditionally (it is the fixed base).

    ``attention_topk_k`` is the per-vertex enumeration cap imposed when
    attention is on (attention disables cycle-batching, so the cycle
    pool must be capped to fit GPU memory — see module note). Default
    8 (fits a 7.6 GiB GPU); raise it to give attention a richer cycle
    pool when memory allows.
    """
    env: dict[str, str] = {}
    if "mixed_tuples" in structure:
        env["HSIKAN_MIXED_TUPLES"] = str(structure["mixed_tuples"])
    if "attention_kind" in structure:
        kind = str(structure["attention_kind"])
        env["HSIKAN_ATTENTION_M_E"] = kind
        if kind in _ATTENTION_KINDS:
            # Attention disables cycle-batching → cap enumeration so the
            # branch fits in GPU memory (see module note).
            env["HSIKAN_TOPK_MODE"] = ATTENTION_TOPK_MODE
            env["HSIKAN_TOPK_K"] = str(attention_topk_k)
    if "per_edge_gate" in structure:
        env["HSIKAN_PER_EDGE_GATE"] = "1" if structure["per_edge_gate"] else "0"
    if "direct_messaging" in structure:
        env["HSIKAN_DIRECT_MESSAGING"] = "1" if structure["direct_messaging"] else "0"
    cli: dict[str, Any] = {}
    if "hidden" in structure:
        cli["hidden"] = int(structure["hidden"])
    if "n_epochs" in structure:
        cli["n_epochs"] = int(structure["n_epochs"])
    return env, cli


def _gib(kilobytes: int) -> float:
    """ru_maxrss is in KiB on Linux → GiB."""
    return kilobytes / (1024.0 * 1024.0)


def run_cell(
    repo: Path,
    *,
    dataset: str,
    seed: int,
    env_patch: dict[str, str],
    hidden: int,
    n_epochs: int,
    log_path: Path,
) -> tuple[dict[str, Any], float, float]:
    """Launch one ``cell_signed_graph`` BA run; return (row, wall_s, peak_rss_gib).

    Per-cell peak RSS is measured with ``/usr/bin/time -v`` (GNU time)
    around the child — reliable across many sequential runs, unlike
    ``getrusage(RUSAGE_CHILDREN).ru_maxrss`` which is a monotonic
    high-water over all reaped children. The result row is the last JSON
    line ``run_final_cell`` prints.
    """
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env.update(env_patch)
    inner = [
        sys.executable, "-m", RUN_CELL_MODULE,
        "--dataset", dataset, "--model", "HSiKAN",
        "--hidden", str(hidden), "--n-epochs", str(n_epochs),
        "--seed", str(seed),
    ]
    gnu_time = Path("/usr/bin/time")
    cmd = ([str(gnu_time), "-v", *inner] if gnu_time.is_file() else inner)
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(repo), env=env, check=False
    )
    wall_s = time.monotonic() - t0
    peak_rss_gib = _parse_time_v_rss_gib(proc.stderr)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_final_cell failed rc={proc.returncode}; see {log_path}"
        )
    row = _last_json_line(proc.stdout)
    return row, wall_s, peak_rss_gib


def _parse_time_v_rss_gib(stderr: str) -> float:
    """Extract peak RSS (GiB) from a ``/usr/bin/time -v`` stderr block.

    Returns ``nan`` if the line is absent (e.g. GNU time unavailable).
    """
    for line in stderr.splitlines():
        if "Maximum resident set size" in line:
            kb = float(line.rsplit(":", 1)[1].strip())
            return _gib(int(kb))
    return float("nan")


def _last_json_line(stdout: str) -> dict[str, Any]:
    """Parse the final JSON object printed on stdout (the result row)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("no JSON result row found on run_final_cell stdout")


def check_budget(
    *,
    auc: Any,
    wall_s: float,
    peak_rss_gib: float,
    mem_cap_gib: float,
    wall_cap_s: float,
) -> list[str]:
    """Return a list of budget-violation messages (empty == pass).

    Pure function so the gate can be unit-tested without a GPU run.
    AUC must be a finite number in ``[0, 1]``; RSS and wall must be
    within their caps.
    """
    violations: list[str] = []
    if peak_rss_gib > mem_cap_gib:
        violations.append(f"peak RSS {peak_rss_gib:.2f} > {mem_cap_gib} GiB")
    if wall_s > wall_cap_s:
        violations.append(f"wall {wall_s:.1f} > {wall_cap_s} s")
    try:
        auc_f = float(auc)
    except (TypeError, ValueError):
        violations.append(f"AUC {auc!r} is not numeric")
    else:
        if not (0.0 <= auc_f <= 1.0):
            violations.append(f"AUC {auc_f} not in [0,1]")
    return violations


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override the train-length unit's n_epochs (abbreviated smoke).",
    )
    ap.add_argument("--hidden", type=int, default=None, help="Override hidden width.")
    ap.add_argument("--mem-cap-gib", type=float, default=7.0)
    ap.add_argument("--wall-cap-s", type=float, default=180.0)
    ap.add_argument("--results-file", default=None)
    ap.add_argument("--log-dir", default="/tmp/hsikan_composite_smoke")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    repo = _repo_root()
    sweep = repo / args.sweep

    solved = solve_regime(repo, sweep, args.regime)
    abb = solved.get("abb") or {}
    units = list(abb.get("units", []))
    if not units:
        print(
            f"[composite-smoke] EMPTY ABB unit set under regime "
            f"{args.regime!r} — the empty-structure regression the "
            f"Pimentel fix addressed. Aborting.",
            file=sys.stderr,
        )
        return 2
    structure = merge_structure_knobs(units)
    env_patch, cli = structure_to_env(structure)
    hidden = args.hidden if args.hidden is not None else int(cli.get("hidden", 16))
    n_epochs = args.n_epochs if args.n_epochs is not None else int(cli.get("n_epochs", 10))

    provenance = {
        "regime": args.regime,
        "abb_units": units,
        "abb_cost": abb.get("cost"),
        "msg_units": solved.get("msg_units"),
        "structure": structure,
        "env_patch": env_patch,
        "hidden": hidden,
        "n_epochs": n_epochs,
        "dataset": args.dataset,
        "seed": args.seed,
        "git_sha": _git_sha(repo),
    }

    if args.dry_run:
        print(json.dumps(provenance, indent=2))
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cell_{args.dataset}_seed{args.seed}.log"

    row, wall_s, peak_rss_gib = run_cell(
        repo,
        dataset=args.dataset,
        seed=args.seed,
        env_patch=env_patch,
        hidden=hidden,
        n_epochs=n_epochs,
        log_path=log_path,
    )

    auc = row.get("auc", row.get("test_auc"))
    result = {
        **provenance,
        "auc": auc,
        "params": row.get("params"),
        "wall_s": round(wall_s, 2),
        "peak_rss_gib": round(peak_rss_gib, 3),
        "result_row": row,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "result_row"}, indent=2))

    if args.results_file:
        with open(args.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # --- Performance-budget gate (numerical assertions, not prints). ---
    violations = check_budget(
        auc=auc,
        wall_s=wall_s,
        peak_rss_gib=peak_rss_gib,
        mem_cap_gib=args.mem_cap_gib,
        wall_cap_s=args.wall_cap_s,
    )
    if violations:
        print("[composite-smoke] BUDGET VIOLATION: " + "; ".join(violations), file=sys.stderr)
        return 1
    print(f"[composite-smoke] OK auc={auc} wall={wall_s:.1f}s rss={peak_rss_gib:.2f}GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
