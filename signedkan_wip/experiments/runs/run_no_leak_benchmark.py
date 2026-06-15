"""No-leakage benchmark (E1) — train-only strict protocol + label-shuffle audit.

Plan: ``docs/plans/2026-06-11-no-leakage-structural-benchmark/``. Each cell
trains a model under the **train-only strict** protocol (cycle pool enumerated
over training edges only, so test-edge signs never enter a cycle — and arity-2,
the trivial edge-readout leak, is excluded) and its **label-shuffled** control.

The shuffle gate (the linchpin): a *clean* model's shuffled AUC must fall to
chance (``<= 0.55``); a model that stays high under shuffle is reading the label
through structure, not learning it — flag it, do not report it as a win.

Each cell is delegated to an existing **strict** runner as a subprocess (fresh
process per cell for RSS isolation; no training/metric logic reimplemented,
CLAUDE.md §6.1). ``run_gomb_smoke`` is the structural-prior strict path
(shuffled → chance, confirmed 2026-06-11: bitcoin_alpha real 0.892 / shuffled
0.503); ``run_sgcn_baseline`` is the same-protocol GNN baseline. The earlier
``cell_signed_graph`` path was rejected by the smoke: it enumerates over the
*full* graph (transductive) and leaked at 0.73 shuffled.

One file, mode arg (§6.5 #13):
    python -m signedkan_wip.experiments.runs.run_no_leak_benchmark --smoke
    python -m signedkan_wip.experiments.runs.run_no_leak_benchmark --full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from signedkan_wip.demos.seminar.resources import fmt_bytes, peak_rss_bytes

# Plan thresholds. A cell is admissible only if its shuffled control is at
# chance; the real run must clear it by a real margin to claim learned signal.
SHUFFLE_GATE = 0.55
MIN_SIGNAL = 0.10
RSS_CAP_BYTES = 16 * 1024 ** 3

# Subprocess strict runners. Each method differs only in (module, auc_key, the
# method-specific CLI tail) — captured in one ``Runner`` so the dispatch in
# ``_run_subprocess`` stays single-pathed (§6.5 #9, no per-method cmd ladder).
@dataclass(frozen=True)
class Runner:
    """A subprocess strict runner: module, its JSON AUC key, and the
    method-specific argv tail built from a :class:`Cell`."""
    module: str
    auc_key: str
    arg_builder: Callable[["Cell"], list[str]]


_BASELINE_AUDIT = "signedkan_wip.experiments.runs.run_baseline_audit"
# Phase B baselines (Table 1 of the Nature paper): all seven route through the
# ONE unified audit runner (§6.5 #1/#13), selected by ``--model``. Width/depth
# are inert here — each method uses its own ``default_hparams``.
AUDIT_METHODS = ("sgcn", "sigat", "sgt", "sgcl", "sigformer", "sesgformer", "dadsgnn")

SUBPROC_RUNNERS: dict[str, Runner] = {
    # The structural prior via run_gomb_smoke (train-only cycle pool, audit-clean):
    # width -> --d-middle, depth -> --middle-n-layers (StackedMiddleHSiKAN at depth>=2).
    "gomb": Runner(
        "signedkan_wip.experiments.runs.run_gomb_smoke", "val_auroc",
        lambda c: ["--d-middle", str(c.width), "--middle-n-layers", str(c.depth)],
    ),
    # Paper-grade: full n_epochs, NO baked-in patience. Measured 2026-06-14 these
    # models keep improving late (sgcl best_epoch 185/200), so an aggressive
    # patience undertrains a baseline — an integrity risk for the audit (an
    # undertrained baseline reads as falsely "clean"). The --patience knob stays
    # available on the CLI for exploratory runs; the Table-1 grid does not use it.
    # Wall driver: the python-attention transformers (sigformer/sesgformer) at
    # ~23 s/epoch on Epinions → ~75 min/arm; see the report's grid-cost section.
    **{
        m: Runner(_BASELINE_AUDIT, "test_auroc",
                  (lambda mm: lambda c: ["--model", mm,
                                         "--reachability", c.reachability])(m))
        for m in AUDIT_METHODS
    },
}
# In-process baselines: SGCN is strict by construction (train-only adjacency, no
# cycle pool to leak) and runs via cell_signed_graph, which exposes
# shuffle_train_signs. The gate VERIFIES its cleanliness; it is not assumed.
INPROC_MODELS = {"SGCN"}

# Pinned per-model architecture as ``(width, depth)`` (plan §"narrow-deep",
# 2026-06-11). The structural prior runs the determinability-per-parameter
# Pareto point h=16, L=8 (reports/2026-05-30-hsikan-depth-narrow-pareto.md):
# narrow-deep beats wide-shallow at ~13 % of the params with ~5× lower seed
# variance — exactly the strict axis E1 measures, so the shallow L=2 generic
# would bias H1 against the prior. For Gömb, ``width`` is the middle HSiKAN
# width (``d_middle``) and ``depth`` the StackedMiddleHSiKAN layer count
# (``middle_n_layers``). SGCN's depth is fixed inside ``cell_signed_graph``;
# only its width (the phase8 panel hidden=32) is set here, depth is inert.
GOMB_NARROW_DEEP = (16, 8)
SGCN_PANEL = (32, 1)


@dataclass(frozen=True)
class Cell:
    """One (model, dataset) configuration with its pinned architecture.

    ``width`` is the hidden dim (``d_middle`` for Gömb, ``hidden`` for SGCN);
    ``depth`` is the stacked-middle layer count (Gömb only — SGCN's depth is
    fixed inside ``cell_signed_graph``, so ``depth`` is inert for it).
    """
    model: str
    dataset: str
    n_epochs: int
    width: int
    depth: int
    reachability: str = "strict"  # message-passing closure: strict|topo|full

    @classmethod
    def make(cls, model: str, dataset: str, n_epochs: int) -> "Cell":
        """Build a cell with the pinned narrow-deep (Gömb) / panel (SGCN) arch."""
        width, depth = GOMB_NARROW_DEEP if model not in INPROC_MODELS else SGCN_PANEL
        return cls(model, dataset, n_epochs, width=width, depth=depth)


# Smoke: structural prior + the same-protocol baseline on one graph, each real
# + shuffle — validates both dispatch paths and that the gate is measurable.
SMOKE_CELLS = (
    Cell.make("gomb", "bitcoin_alpha", n_epochs=60),
    Cell.make("SGCN", "bitcoin_alpha", n_epochs=120),
)

# Full E1: the structural prior (narrow-deep) vs the same-protocol baseline,
# both Bitcoin graphs. (SiGAT-strict is a follow-up; SGCN keys an existing
# strict runner.)
FULL_CELLS = tuple(
    Cell.make(model, dataset, n_epochs=200)
    for dataset in ("bitcoin_alpha", "bitcoin_otc")
    for model in ("gomb", "SGCN")
)

# Phase B baseline audit grid (Nature Table 1): the seven baselines × five
# datasets, each real + shuffle, 5 seeds. Width/depth inert — every method uses
# its own ``default_hparams`` inside run_baseline_audit. Run via ``--baselines``;
# resumable per (model, dataset, seed, shuffle) like the E1 grid.
AUDIT_DATASETS = ("bitcoin_alpha", "bitcoin_otc", "slashdot", "reddit_body", "epinions")
BASELINE_CELLS = tuple(
    Cell(model=model, dataset=dataset, n_epochs=200, width=32, depth=0)
    for dataset in AUDIT_DATASETS
    for model in AUDIT_METHODS
)


def _parse_result(stdout: str, auc_key: str) -> tuple[float | None, int | None]:
    """Last JSON-object line of a strict runner's stdout → (auc, n_params)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            return d.get(auc_key), d.get("n_params")
    return None, None


def _run_subprocess(cell: Cell, seed: int, shuffle: bool) -> tuple[float | None, int | None]:
    """Drive a strict CLI runner; return (auc, n_params) from its JSON line.

    The method-specific argv tail comes from the runner's ``arg_builder`` so this
    dispatch is identical for the structural prior and every baseline.
    """
    runner = SUBPROC_RUNNERS[cell.model]
    cmd = [sys.executable, "-m", runner.module, "--dataset", cell.dataset,
           "--seed", str(seed), "--n-epochs", str(cell.n_epochs),
           *runner.arg_builder(cell)]
    if shuffle:
        cmd.append("--shuffle-train-signs")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    auc, n_params = _parse_result(proc.stdout, runner.auc_key)
    if auc is None:
        print(f"  [warn] no JSON from {cell.model}/{cell.dataset} "
              f"(exit {proc.returncode}); stderr tail:\n{proc.stderr[-400:]}")
    return auc, n_params


def _run_inprocess(cell: Cell, seed: int, shuffle: bool) -> tuple[float | None, int | None]:
    """SGCN baseline via cell_signed_graph (strict-by-construction adjacency)."""
    from .run_final_cell import cell_signed_graph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = cell_signed_graph(cell.dataset, cell.model, hidden=cell.width,
                          n_epochs=cell.n_epochs, max_k4=20_000,
                          device=device, seed=seed, shuffle_train_signs=shuffle)
    return r.get("auc"), r.get("n_params")


def run_cell(cell: Cell, seed: int, shuffle: bool) -> dict[str, Any]:
    """Train+eval one cell via its strict path (real or shuffled).

    Preconditions: ``cell.model`` keys SUBPROC_RUNNERS or is in INPROC_MODELS.
    Postconditions: row carries ``auc`` (held-out, None on runner failure).
    """
    t0 = time.perf_counter()
    if cell.model in INPROC_MODELS:
        auc, n_params = _run_inprocess(cell, seed, shuffle)
    else:
        auc, n_params = _run_subprocess(cell, seed, shuffle)
    return dict(dataset=cell.dataset, model=cell.model, seed=seed,
                shuffle=shuffle, auc=auc, n_params=n_params,
                width=cell.width, depth=cell.depth,
                reachability=cell.reachability,
                wall_s=time.perf_counter() - t0)


def _result_key(row: dict[str, Any]) -> tuple[str, str, int, bool, str]:
    """Resume identity of a result row: (dataset, model, seed, shuffle, reachability).

    Tolerates legacy rows written before the ``seed``/``reachability`` fields
    existed (treated as seed 0 / "strict"), so an older JSONL still resumes
    cleanly and never collides with new topo/full rows.
    """
    return (row["dataset"], row["model"],
            int(row.get("seed", 0)), bool(row["shuffle"]),
            row.get("reachability", "strict"))


def _load_done(out_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, int, bool]]]:
    """Existing rows + their resume keys, so completed cells are skipped."""
    if not out_path.exists():
        return [], set()
    rows = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    return rows, {_result_key(r) for r in rows}


def _append_row(out_path: Path, row: dict[str, Any]) -> None:
    """Append one result as a JSONL line — the per-cell checkpoint."""
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def audit_gate(real_auc: float | None, shuf_auc: float | None) -> tuple[bool, bool]:
    """(clean, has_signal): clean = shuffled at chance; has_signal = real clears it.

    A leaky model fails ``clean`` (stays high under shuffle, like the optuna-c2
    config at 0.99). A model with no learned signal fails ``has_signal``.
    """
    clean = shuf_auc is not None and shuf_auc <= SHUFFLE_GATE
    has_signal = (real_auc is not None and shuf_auc is not None
                  and (real_auc - shuf_auc) >= MIN_SIGNAL)
    return clean, has_signal


def _run_cell_seed(
    cell: Cell, seed: int, out_path: Path,
    rows: list[dict[str, Any]], done: set[tuple[str, str, int, bool]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Real + shuffled run for one (cell, seed), resumable per shuffle arm.

    Each arm already present in ``done`` is reused from ``rows``; a fresh arm is
    run and checkpointed (appended) immediately, so a crash loses at most the
    in-flight arm. Returns the (real, shuffled) rows.
    """
    arms: dict[bool, dict[str, Any]] = {}
    for shuffle in (False, True):
        key = (cell.dataset, cell.model, seed, shuffle)
        if key in done:
            arms[shuffle] = next(r for r in rows if _result_key(r) == key)
            continue
        row = run_cell(cell, seed, shuffle=shuffle)
        _append_row(out_path, row)
        rows.append(row)
        done.add(key)
        arms[shuffle] = row
    return arms[False], arms[True]


def main(
    smoke: bool = True, out_path: Any = None, seeds: tuple[int, ...] = (0,),
    cells: tuple[Cell, ...] | None = None, tag: str | None = None,
) -> list[dict[str, Any]]:
    """Run cells × seeds real+shuffled, apply the gate, checkpoint JSONL, report.

    Resumable: an existing ``out_path`` is loaded and completed (cell, seed,
    shuffle) arms are skipped, so an interrupted grid continues where it stopped.
    ``cells`` overrides the smoke/E1 default (used by the ``--baselines`` grid).
    """
    cells = cells if cells is not None else (SMOKE_CELLS if smoke else FULL_CELLS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = tag or ("smoke" if smoke else "e1")
    out_path = Path(out_path) if out_path else Path(
        f"signedkan_wip/experiments/results/no_leak_{label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, done = _load_done(out_path)
    print(f"No-leakage {label}: {len(cells)} cells × "
          f"{len(seeds)} seeds (strict train-only pool) on {device}"
          + (f"  [resume: {len(done)} arms done]" if done else ""))

    wall_t0 = time.perf_counter()
    for cell in cells:
        for seed in seeds:
            real, shuf = _run_cell_seed(cell, seed, out_path, rows, done)
            clean, has_signal = audit_gate(real["auc"], shuf["auc"])
            verdict = ("CLEAN+SIGNAL" if clean and has_signal
                       else "LEAKS" if not clean
                       else "NO-SIGNAL")
            ra = "None" if real["auc"] is None else f"{real['auc']:.4f}"
            sa = "None" if shuf["auc"] is None else f"{shuf['auc']:.4f}"
            print(f"  {cell.model:<7} {cell.dataset:<14} seed={seed} "
                  f"real={ra} shuffled={sa}  -> {verdict}")

    peak = peak_rss_bytes()
    wall = time.perf_counter() - wall_t0
    print(f"\nWrote {out_path} ({len(rows)} rows)\n"
          f"Peak RSS: {fmt_bytes(peak)}  |  wall: {wall:.1f}s")
    assert peak is None or peak < RSS_CAP_BYTES, \
        f"peak RSS {fmt_bytes(peak)} exceeds 16 GB cap"
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="no-leakage strict + shuffle audit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="1 cell, validate path")
    mode.add_argument("--full", action="store_true", help="full E1 grid")
    mode.add_argument("--baselines", action="store_true",
                      help="Phase B Table-1 grid: 7 baselines × 5 datasets")
    parser.add_argument(
        "--seeds", default=None,
        help="comma-separated seeds; default 0 (smoke) / 0,1,2,3,4 (full/baselines).")
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of AUDIT_DATASETS (--baselines only).")
    parser.add_argument("--models", default=None,
                        help="comma-separated subset of AUDIT_METHODS (--baselines only).")
    parser.add_argument("--reachability", default="strict",
                        choices=["strict", "topo", "full"],
                        help="message-passing closure for --baselines (default strict).")
    parser.add_argument("--out", default=None)
    a = parser.parse_args()
    if a.seeds:
        seeds = tuple(int(s) for s in a.seeds.split(",") if s.strip())
    else:
        seeds = tuple(range(5)) if (a.full or a.baselines) else (0,)

    if a.baselines:
        ds = a.datasets.split(",") if a.datasets else AUDIT_DATASETS
        ms = a.models.split(",") if a.models else AUDIT_METHODS
        cells = tuple(Cell(model=m, dataset=d, n_epochs=200, width=32, depth=0,
                           reachability=a.reachability)
                      for d in ds for m in ms)
        main(smoke=False, out_path=a.out, seeds=seeds, cells=cells,
             tag=f"baselines_{a.reachability}")
    else:
        main(smoke=not a.full, out_path=a.out, seeds=seeds)
