"""HyMeKo-driven HSiKAN training cell.

Reads a HyMeKo description (architecture + training + optional
sweep) and runs an actual training cell. Two execution modes:

- **single**: parse arch + training .hymeko, instantiate a config,
  invoke ``run_final_cell.cell_signed_graph`` once.
- **sweep**: parse a sweep .hymeko, enumerate the search space (grid
  / GA / MSG-axiom-feasibility), run one cell per point, emit JSONL.

Usage:

    # Single training cell from a HyMeKo description:
    python -m signedkan_wip.src.hymeko_driver \\
        --arch     data/hsikan/arch_mixed_k34.hymeko \\
        --training data/hsikan/training.hymeko \\
        --dataset  bitcoin_alpha

    # Grid sweep:
    python -m signedkan_wip.src.hymeko_driver \\
        --sweep data/hsikan/sweep_grid.hymeko \\
        --dataset bitcoin_alpha

    # P-graph axiom-feasibility sweep (MSG/SSG/ABB):
    python -m signedkan_wip.src.hymeko_driver \\
        --sweep data/hsikan/sweep_msg.hymeko

    # Gömb smoke (subprocess ``run_gomb_smoke``) — default training profile:
    python -m signedkan_wip.src.hymeko_driver \\
        --backend gomb --device cpu

    # Gömb grid sweep over ``gomb.topk`` (see ``data/hsikan/sweep_grid_gomb.hymeko``):
    python -m signedkan_wip.src.hymeko_driver \\
        --backend gomb --sweep data/hsikan/sweep_grid_gomb.hymeko --device cpu

The driver maps tagged hyperedges to existing pipeline knobs
(`HSIKAN_TOPK_*` env vars + `cell_signed_graph` arguments) so the
first version can stand on the shoulders of the existing training
code rather than re-implementing forward/backward.

Schema convention (see data/hsikan/*.hymeko):
- Tensors: nodes tagged `<input>`, `<activation>`, `<output>`.
- Layers: nodes inheriting from a layer-class base (signedkan_layer,
  arity_mixer, signed_classifier, walk_layer); sub-tags drive kwargs.
- Dataflow: `@name <dataflow> { (+in, ~layer, -out); }`.
- Training step: `@name <forward|loss|backward|optimizer|...>` tag.
- Sweep range: `@name <param_range, target="x.y"> { values [...]; }`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

# Shared HyMeKo-IR helpers — single source of truth lives in hymeko_ir.py.
from .hymeko_ir import (
    read_hymeko as _read_hymeko,
    all_items as _all_items,
    has_tag as _has_tag,
    has_base as _has_base,
)


def _child_value(item: dict, child_name: str, default=None):
    """Find a child node by name and return its scalar value.

    NOTE: behaves slightly differently from hymeko_ir.child_value: this
    variant does NOT strip surrounding quotes from string values; the
    driver does its own ``.strip('"')`` at each call site for backward
    compatibility with existing parse_training logic.  When migrating a
    call site, prefer hymeko_ir.child_value (which strips)."""
    body = item.get("body") or []
    for c in body:
        if c.get("kind") == "node" and c.get("name") == child_name:
            return c.get("value", default)
    return default


def _tag_value(item: dict, key: str, default=None):
    """Read a `key="value"` style tag-pair from the item's value
    field. The HyMeKo grammar doesn't carry per-tag values directly,
    but the convention is: edge value carries `key="value"`-shaped
    string OR cost number; for kwargs we use child statements
    (see _child_value)."""
    return _child_value(item, key, default)


# ─── Architecture .hymeko → MixedAritySignedKANConfig ──────────────


# parse_arch lives in hymeko_ir; re-exported here for backward compat
# with callers that ``from .hymeko_driver import parse_arch``.
from .hymeko_ir import parse_arch as _ir_parse_arch


def parse_arch(arch_path: str) -> dict:
    """Backward-compat wrapper over hymeko_ir.parse_arch.

    Returns the same shape but with ``arities`` as a list (legacy
    convention) — the IR helper returns a tuple."""
    cfg = _ir_parse_arch(arch_path)
    return {**cfg, "arities": list(cfg["arities"])}


# ─── Training .hymeko → run knobs + env vars ───────────────────────


def parse_training(training_path: str) -> dict:
    """Walk a training .hymeko and return a dict of knobs the driver
    routes into HSIKAN_TOPK_* env vars + run_final_cell kwargs."""
    tree = _read_hymeko(training_path)
    items = _all_items(tree)
    # Constants
    consts = {c["name"]: c["value"] for c in tree.get("consts") or []}

    knobs: dict[str, Any] = {
        "n_epochs": int(consts.get("N_EPOCHS", 30)),
        "lr": float(consts.get("LR", 0.01)),
        "weight_decay": float(consts.get("WD", 0.0001)),
        "seed": int(consts.get("SEED", 0)),
        # Defaults (overwritten by hyperedges below)
        "topk_mode": "",
        "topk_k": 16,
        "topk_scorer": "fraction_negative",
        "topk_pruner": "none",
        "loss_kind": "bce",
        "entropy_lambda": 0.0,
        "entropy_kind": "spectral",
        "dataset_name": None,
        "max_k4": 200000,
        # Gömb smoke (`--backend gomb` + `run_gomb_msg_sweep` base profile)
        "gomb_topk": 48,
        "gomb_cycle_abb_mode": "none",
        "gomb_cycle_abb_fullness_gate": 0.25,
        "gomb_d_embed": 16,
        "gomb_d_outer": 8,
        "gomb_M_outer": 2,
        "gomb_d_middle": 16,
        "gomb_d_core": 16,
        "gomb_k": 3,
    }

    for it in items:
        if it["kind"] != "edge":
            continue
        tags = it.get("tags") or []

        # @load_dataset <dataset, name="...">
        if "dataset" in tags:
            # Try to find a `name` tag-value via child node first
            for child in it.get("body", []) or []:
                if child.get("kind") == "node" and child.get("name") == "name":
                    v = child.get("value")
                    if isinstance(v, str):
                        knobs["dataset_name"] = v.strip('"')
            # Fallback: edge value
            v = it.get("value")
            if isinstance(v, str) and not knobs["dataset_name"]:
                knobs["dataset_name"] = v.strip('"')

        # @enumerate_cycles <cycle_enum>
        if "cycle_enum" in tags:
            mode = _child_value(it, "mode", "")
            if isinstance(mode, str):
                knobs["topk_mode"] = mode.strip('"')
            knobs["topk_k"] = int(_child_value(it, "m_per_vertex", knobs["topk_k"]))
            scorer = _child_value(it, "scorer", knobs["topk_scorer"])
            if isinstance(scorer, str):
                knobs["topk_scorer"] = scorer.strip('"')
            pruner = _child_value(it, "pruner", knobs["topk_pruner"])
            if isinstance(pruner, str):
                knobs["topk_pruner"] = pruner.strip('"')

        # @compute_loss <loss, kind="bce">
        if "loss" in tags:
            loss_kind = _child_value(it, "loss_kind", knobs["loss_kind"])
            if isinstance(loss_kind, str):
                knobs["loss_kind"] = loss_kind.strip('"')
            knobs["entropy_lambda"] = float(
                _child_value(it, "entropy_lambda", knobs["entropy_lambda"]))
            ek = _child_value(it, "entropy_kind", knobs["entropy_kind"])
            if isinstance(ek, str):
                knobs["entropy_kind"] = ek.strip('"')

        # @optimizer_step <optimizer>
        if "optimizer" in tags:
            knobs["lr"] = float(_child_value(it, "lr", knobs["lr"]))
            knobs["weight_decay"] = float(
                _child_value(it, "weight_decay", knobs["weight_decay"]))

        # @train_loop <epoch_loop>
        if "epoch_loop" in tags:
            knobs["n_epochs"] = int(_child_value(it, "n_epochs", knobs["n_epochs"]))

        # @gomb_smoke <gomb_smoke> { topk …; cycle_abb_mode …; … }
        if "gomb_smoke" in tags:
            tv = _child_value(it, "topk", None)
            if tv is not None:
                knobs["gomb_topk"] = int(tv)
            cm = _child_value(it, "cycle_abb_mode", None)
            if isinstance(cm, str) and cm.strip():
                knobs["gomb_cycle_abb_mode"] = cm.strip().strip('"')
            gv = _child_value(it, "cycle_abb_fullness_gate", None)
            if gv is not None:
                knobs["gomb_cycle_abb_fullness_gate"] = float(gv)
            for key, dest, caster in (
                ("d_embed", "gomb_d_embed", int),
                ("d_outer", "gomb_d_outer", int),
                ("M_outer", "gomb_M_outer", int),
                ("d_middle", "gomb_d_middle", int),
                ("d_core", "gomb_d_core", int),
                ("k", "gomb_k", int),
            ):
                v = _child_value(it, key, None)
                if v is not None:
                    knobs[dest] = caster(v)

    return knobs


# ─── Sweep .hymeko → list of config dicts ──────────────────────────


def parse_sweep(sweep_path: str) -> dict:
    """Parse a sweep .hymeko (grid / GA / MSG)."""
    tree = _read_hymeko(sweep_path)
    items = _all_items(tree)
    ranges: dict[str, list] = {}
    policy: dict[str, Any] = {}
    for it in items:
        tags = it.get("tags") or []
        if it["kind"] != "edge":
            continue
        if "param_range" in tags:
            target = _tag_value(it, "target")
            # `target` may live in tags or as a child node — we
            # accept either.  Fallback to edge name.
            if not target:
                # Extract from tag-payload-style tags ('target="..."')
                for t in tags:
                    if t.startswith("target="):
                        target = t.split("=", 1)[1].strip('"')
            target = target or it.get("name")
            values = _child_value(it, "values", []) or []
            ranges[target] = values
        elif "sweep_policy" in tags:
            for child in it.get("body") or []:
                if child.get("kind") == "node":
                    policy[child["name"]] = child.get("value")
    return {
        "ranges": ranges,
        "policy": policy,
        "name": tree.get("name", "Sweep"),
    }


# ─── Execution: HyMeKo config → existing run_final_cell ───────────


def run_single(
    arch: dict,
    training: dict,
    dataset: str | None,
    hidden_override: int | None = None,
    *,
    backend: str = "hsikan",
    device: str | None = None,
) -> dict:
    """Run one training cell with the parsed arch+training config.
    Routes through `signedkan_wip.src.run_final_cell.cell_signed_graph`,
    wiring HSIKAN_TOPK_* env vars on the way in.

    When ``backend == "gomb"``, runs ``run_gomb_smoke`` in a subprocess and
    returns the trailing JSON summary (plus ``backend`` / ``device`` keys).
    """
    if backend == "gomb":
        return run_single_gomb(training, dataset, device=device)

    import torch

    ds = dataset or training.get("dataset_name") or "bitcoin_alpha"
    hidden = hidden_override if hidden_override is not None else arch["hidden"]
    n_epochs = training["n_epochs"]
    seed = training["seed"]
    max_k4 = training.get("max_k4", 200000)

    # Set env vars BEFORE importing the inner training code (they are
    # consumed at cycle-enum time).
    if training.get("topk_mode"):
        os.environ["HSIKAN_TOPK_MODE"] = training["topk_mode"]
        os.environ["HSIKAN_TOPK_K"] = str(training["topk_k"])
        os.environ["HSIKAN_TOPK_SCORER"] = training["topk_scorer"]
        os.environ["HSIKAN_TOPK_PRUNER"] = training["topk_pruner"]
    if training["entropy_lambda"] > 0:
        os.environ["HSIKAN_ENTROPY_LAMBDA"] = str(training["entropy_lambda"])

    # Pin arities the user asked for (mixed_k34 → "3,4")
    if arch.get("arities"):
        os.environ["HSIKAN_ARITIES"] = ",".join(str(a) for a in arch["arities"])

    from signedkan_wip.src.run_final_cell import cell_signed_graph

    if device:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = cell_signed_graph(
        ds, "HSiKAN", hidden, n_epochs, max_k4, dev, seed=seed,
    )
    if out is not None:
        out["seed"] = seed
        out["hymeko_arch"] = arch.get("name")
        out["hymeko_training"] = {
            "topk_mode": training.get("topk_mode"),
            "topk_k": training.get("topk_k"),
            "topk_scorer": training.get("topk_scorer"),
            "topk_pruner": training.get("topk_pruner"),
            "n_epochs": n_epochs,
            "lr": training["lr"],
        }
    return out or {}


def run_single_gomb(
    training: dict,
    dataset: str | None,
    *,
    device: str | None = None,
) -> dict:
    """One Gömb smoke: subprocess ``run_gomb_smoke`` from ``parse_training`` knobs."""
    import subprocess

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    dev = device
    if not dev:
        if torch is not None and torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"

    repo = Path(__file__).resolve().parents[2]
    ds = dataset or training.get("dataset_name") or "sbm_n200"
    base = {**training, "_python_executable": sys.executable}
    from signedkan_wip.src.gomb_pgraph_mapping import smoke_argv_from_knobs

    cmd = smoke_argv_from_knobs(
        dataset=ds,
        device=dev,
        edge_split="80_10_10",
        seed=int(training.get("seed", 0)),
        base=base,
        structure={},
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=3600,
    )
    if proc.returncode != 0:
        return {
            "backend": "gomb",
            "device": dev,
            "dataset": ds,
            "error": (proc.stderr or "")[-4000:],
            "returncode": proc.returncode,
        }
    row = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            row = json.loads(line)
            break
    out = row or {}
    out["backend"] = "gomb"
    out["device"] = dev
    return out


def run_sweep(
    sweep: dict,
    base_arch: dict,
    base_training: dict,
    dataset: str,
    output_path: str | None,
    max_runs: int | None = None,
    *,
    backend: str = "hsikan",
    device: str | None = None,
) -> list[dict]:
    """Grid sweep — Cartesian product of the param_range edges."""
    ranges = sweep["ranges"]
    policy = sweep["policy"]
    if not ranges:
        print("no param_range edges in sweep file", file=sys.stderr)
        return []

    keys = list(ranges.keys())
    grids = [ranges[k] for k in keys]
    cells = list(itertools.product(*grids))
    if max_runs is None:
        max_runs = int(policy.get("max_runs") or len(cells))
    cells = cells[:max_runs]
    print(f"sweep: {len(cells)} cells (capped at {max_runs})", file=sys.stderr)

    results = []
    for ix, point in enumerate(cells):
        cfg = dict(zip(keys, point))
        print(f"\n── cell {ix+1}/{len(cells)}: {cfg}", file=sys.stderr)
        # Apply the cell to base_arch / base_training copies
        arch = dict(base_arch)
        training = dict(base_training)
        if "model.hidden" in cfg:
            arch["hidden"] = int(cfg["model.hidden"])
        if "model.grid" in cfg:
            arch["grid"] = int(cfg["model.grid"])
        if "model.arities" in cfg:
            arch["arities"] = [int(x) for x in cfg["model.arities"]]
        if "topk.m_per_vertex" in cfg:
            training["topk_k"] = int(cfg["topk.m_per_vertex"])
        if "topk.pruner" in cfg:
            v = cfg["topk.pruner"]
            training["topk_pruner"] = v.strip('"') if isinstance(v, str) else v
        if "optimizer.lr" in cfg:
            training["lr"] = float(cfg["optimizer.lr"])
        if "gomb.topk" in cfg:
            training["gomb_topk"] = int(cfg["gomb.topk"])
        if "gomb.cycle_abb_mode" in cfg:
            v = cfg["gomb.cycle_abb_mode"]
            training["gomb_cycle_abb_mode"] = (
                v.strip('"') if isinstance(v, str) else str(v)
            )
        try:
            out = run_single(
                arch, training, dataset, backend=backend, device=device
            )
            out["sweep_cell"] = cfg
            results.append(out)
            if output_path:
                with open(output_path, "a") as f:
                    f.write(json.dumps(out) + "\n")
            print(json.dumps(out), flush=True)
        except Exception as e:
            err = {"sweep_cell": cfg, "error": repr(e)}
            results.append(err)
            print(f"  cell failed: {e}", file=sys.stderr)
    return results


# ─── CLI ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="HyMeKo-driven HSiKAN or Gömb training")
    ap.add_argument(
        "--backend",
        choices=("hsikan", "gomb"),
        default="hsikan",
        help="Execution backend: HSiKAN cell vs Gömb smoke subprocess.",
    )
    ap.add_argument("--arch",     help="Architecture .hymeko")
    ap.add_argument("--training", help="Training .hymeko")
    ap.add_argument("--sweep",    help="Sweep .hymeko (grid / GA / MSG)")
    ap.add_argument("--dataset",  default=None,
                    help="Override the dataset (default: from training file)")
    ap.add_argument(
        "--device",
        default=None,
        help="cpu / cuda (Gömb backend only; default: auto)",
    )
    ap.add_argument("--max-runs", type=int, default=None,
                    help="Cap sweep runs (default from policy)")
    ap.add_argument("--output",   default=None,
                    help="Append per-cell JSON to this file")
    ap.add_argument("--print-only", action="store_true",
                    help="Parse and print parsed config; don't run")
    args = ap.parse_args()

    if args.sweep:
        sweep = parse_sweep(args.sweep)
        # Use the first arch + training in the sweep file's @use_*
        # references; if not present, require explicit --arch/--training
        if args.backend == "gomb":
            arch = {"name": "gomb_placeholder", "hidden": 0, "arities": []}
            tpath = args.training or str(
                Path(__file__).resolve().parents[2]
                / "data"
                / "hsikan"
                / "gomb_training.hymeko"
            )
            training = parse_training(tpath)
        else:
            arch = parse_arch(args.arch) if args.arch else \
                parse_arch("data/hsikan/arch_mixed_k34.hymeko")
            training = parse_training(args.training) if args.training else \
                parse_training("data/hsikan/training.hymeko")
        if args.print_only:
            print(json.dumps({"arch": arch, "training": training,
                              "sweep": sweep}, indent=2, default=str))
            return
        run_sweep(
            sweep,
            arch,
            training,
            args.dataset or training.get("dataset_name") or "bitcoin_alpha",
            args.output,
            max_runs=args.max_runs,
            backend=args.backend,
            device=args.device,
        )
        return

    if args.backend == "gomb":
        tpath = args.training or str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "hsikan"
            / "gomb_training.hymeko"
        )
        training = parse_training(tpath)
        if args.print_only:
            print(json.dumps({"backend": "gomb", "training": training},
                             indent=2, default=str))
            return
        out = run_single_gomb(training, args.dataset, device=args.device)
        print(json.dumps(out))
        return

    if not args.arch or not args.training:
        ap.error("--arch and --training required (or pass --sweep)")
    arch = parse_arch(args.arch)
    training = parse_training(args.training)
    if args.print_only:
        print(json.dumps({"arch": arch, "training": training}, indent=2,
                          default=str))
        return
    out = run_single(arch, training, args.dataset, backend=args.backend)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
