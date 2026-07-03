"""Protocol-matched, honest rotor-line-vs-SiGAT comparison.

Plan: ``docs/plans/2026-06-17-protocol-matched-sigat/``. The 2026-06-17 session
chased a ~0.04 "SiGAT gap"; a provenance check found it was partly a *protocol*
artifact: the rotor-HSiKAN line is scored with ``--dedup`` (true held-out: drop
test edges whose ``(u,v)`` is also a train pair) while the ``sigat_rotor`` audit
target was scored **without** it — strictly easier, ~65 % more (leaky) test edges
on bitcoin_alpha. So target and model used *different* test sets.

This driver measures the comparison **like-for-like**: every model on both
protocols (deduped + non-deduped), 5 seeds, each real + label-shuffled. A number
is reportable only if its shuffled control falls to chance (the shuffle gate,
reused verbatim from :mod:`run_no_leak_benchmark`) — otherwise the model is
reading the label through structure, not learning it.

No training/metric logic is reimplemented (§6.5 #1/#3): audit baselines route
through :func:`run_baseline_audit.run_audit` and the rotor-HSiKAN line through
:func:`run_hsikan_rotor.run`, each wrapped in a Strategy closure so the grid loop
is model-family-agnostic (§6.5 #9, no ``if family ==`` ladder). The honest gap
table is produced by :mod:`hymeko_neuro.experiments.eval.aggregate_protocol_match`.

One file, mode arg (§6.5 #13):
    python -m hymeko_neuro.experiments.runs.run_protocol_matched_sigat --smoke
    python -m hymeko_neuro.experiments.runs.run_protocol_matched_sigat --full
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from hymeko_neuro.demos.seminar.resources import fmt_bytes, peak_rss_bytes
from hymeko_neuro.experiments.runs.run_baseline_audit import run_audit
from hymeko_neuro.experiments.runs.run_hsikan_rotor import TrainConfig, run as run_hsikan
from hymeko_neuro.experiments.runs.run_no_leak_benchmark import (
    RSS_CAP_BYTES,
    audit_gate,
)

# Result-dict producer: (dataset, seed, dedup, shuffle) -> dict with at least
# ``test_auroc`` and ``n_params``. Both backends already emit exactly that.
ResultFn = Callable[[str, int, bool, bool], dict[str, Any]]


@dataclass(frozen=True)
class ModelSpec:
    """One model row of the comparison: a display label + its result producer.

    The closure hides the backend (audit registry vs the rotor driver) so the
    grid loop never branches on model family — adding a model is one tuple entry.
    """
    label: str
    run: ResultFn


def _audit_spec(model: str, n_epochs: int = 120) -> ModelSpec:
    """A registry baseline scored through the unified strict+shuffle audit."""
    def _run(dataset: str, seed: int, dedup: bool, shuffle: bool) -> dict[str, Any]:
        return run_audit(model, dataset, seed, n_epochs=n_epochs,
                         shuffle_train_signs=shuffle, dedup=dedup)
    return ModelSpec(label=model, run=_run)


def _hsikan_rotor_spec() -> ModelSpec:
    """The leakage-free rotor-HSiKAN line at its 2026-06-17 winning recipe:
    signed nlerp rotor propagation r2/sw4, bilinear head, tuned BCE recipe
    (class-weighted, early-stop, wd 1e-4, grad-clip 1.0)."""
    cfg = TrainConfig(weight_decay=1e-4, grad_clip=1.0,
                      class_weight=True, early_stop=True)

    def _run(dataset: str, seed: int, dedup: bool, shuffle: bool) -> dict[str, Any]:
        return run_hsikan(dataset, seed, embed="rotor", shuffle=shuffle,
                          head="bilinear", dedup=dedup,
                          rotor_prop_rounds=2, rotor_prop_self_weight=4.0, cfg=cfg)
    return ModelSpec(label="hsikan_rotor_r2sw4", run=_run)


# The comparison set: pure SiGAT, the two rotor audit baselines, and the
# leakage-free rotor-HSiKAN line. Pure SiGAT anchors the ceiling; sigat_rotor is
# the target the headline gap was measured against; cayley_rotor is the inductive
# rotor embedding; hsikan_rotor is our line.
SPECS: tuple[ModelSpec, ...] = (
    _audit_spec("sigat"),
    _audit_spec("sigat_rotor"),
    _audit_spec("cayley_rotor"),
    _hsikan_rotor_spec(),
)
DATASETS = ("bitcoin_alpha", "bitcoin_otc")
PROTOCOLS = (False, True)  # dedup off (non-deduped) / on (true held-out)


@dataclass(frozen=True)
class Cell:
    """One (model, dataset, protocol) configuration; seeds iterate outside."""
    spec: ModelSpec
    dataset: str
    dedup: bool


def _cells(specs: tuple[ModelSpec, ...], datasets: tuple[str, ...]) -> list[Cell]:
    return [Cell(spec, ds, dedup)
            for spec in specs for ds in datasets for dedup in PROTOCOLS]


def _row_key(row: dict[str, Any]) -> tuple[str, str, bool, int, bool]:
    """Resume identity: (model, dataset, dedup, seed, shuffle)."""
    return (row["model"], row["dataset"], bool(row["dedup"]),
            int(row["seed"]), bool(row["shuffle"]))


def _load_done(
    out_path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, bool, int, bool]]]:
    if not out_path.exists():
        return [], set()
    rows = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    return rows, {_row_key(r) for r in rows}


def run_arm(cell: Cell, seed: int, shuffle: bool) -> dict[str, Any]:
    """Train+eval one arm; normalise the backend result into a comparison row.

    Preconditions: ``cell.dataset`` loads; ``seed >= 0``.
    Postconditions: row carries ``auc`` (``test_auroc``, may be ``nan`` on a
    one-class/empty held-out slice) and ``n_params``; ``wall_s`` is wall time.
    """
    t0 = time.perf_counter()
    r = cell.spec.run(cell.dataset, seed, cell.dedup, shuffle)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release per-cell device tensors; RSS stays flat
    return dict(model=cell.spec.label, dataset=cell.dataset, dedup=cell.dedup,
                seed=seed, shuffle=shuffle,
                auc=r.get("test_auroc"), n_params=r.get("n_params"),
                n_test=r.get("n_test"), n_test_dropped=r.get("n_test_dropped"),
                wall_s=round(time.perf_counter() - t0, 2))


def main(smoke: bool = True, out_path: Any = None,
         seeds: tuple[int, ...] | None = None,
         specs: tuple[ModelSpec, ...] | None = None,
         datasets: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Run the model×dataset×protocol grid × seeds, each real+shuffle, gated.

    Resumable: completed (model, dataset, dedup, seed, shuffle) arms in an
    existing ``out_path`` are skipped, so an interrupted grid continues.
    """
    if smoke:
        specs = specs or (_audit_spec("sigat"),)
        datasets = datasets or ("bitcoin_alpha",)
        seeds = seeds or (0,)
    else:
        specs = specs or SPECS
        datasets = datasets or DATASETS
        seeds = seeds or (0, 1, 2, 3, 4)

    cells = _cells(specs, datasets)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = "smoke" if smoke else "full"
    out_path = Path(out_path) if out_path else Path(
        f"hymeko_neuro/experiments/results/protocol_matched_{label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, done = _load_done(out_path)
    print(f"Protocol-matched comparison [{label}]: {len(cells)} cells × "
          f"{len(seeds)} seeds (real+shuffle) on {device}"
          + (f"  [resume: {len(done)} arms done]" if done else ""))

    wall_t0 = time.perf_counter()
    for cell in cells:
        for seed in seeds:
            arms: dict[bool, dict[str, Any]] = {}
            for shuffle in (False, True):
                key = (cell.spec.label, cell.dataset, cell.dedup, seed, shuffle)
                hit = next((r for r in rows if _row_key(r) == key), None)
                if hit is None:
                    hit = run_arm(cell, seed, shuffle)
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(hit) + "\n")
                    rows.append(hit)
                    done.add(key)
                arms[shuffle] = hit
            clean, has_signal = audit_gate(arms[False]["auc"], arms[True]["auc"])
            verdict = ("CLEAN+SIGNAL" if clean and has_signal
                       else "LEAKS" if not clean else "NO-SIGNAL")
            proto = "dedup" if cell.dedup else "raw  "
            ra = "  nan" if arms[False]["auc"] is None else f"{arms[False]['auc']:.4f}"
            sa = "  nan" if arms[True]["auc"] is None else f"{arms[True]['auc']:.4f}"
            print(f"  {cell.spec.label:<18} {cell.dataset:<14} {proto} seed={seed} "
                  f"real={ra} shuf={sa} -> {verdict}")

    peak = peak_rss_bytes()
    wall = time.perf_counter() - wall_t0
    print(f"\nWrote {out_path} ({len(rows)} rows)\n"
          f"Peak RSS: {fmt_bytes(peak)}  |  wall: {wall:.1f}s")
    assert peak is None or peak < RSS_CAP_BYTES, \
        f"peak RSS {fmt_bytes(peak)} exceeds 16 GB cap"
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="1 model × 1 dataset × both protocols × 1 seed")
    mode.add_argument("--full", action="store_true",
                      help="all models × both Bitcoin graphs × both protocols × 5 seeds")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (default 0 smoke / 0,1,2,3,4 full)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = (tuple(int(s) for s in a.seeds.split(",") if s.strip())
             if a.seeds else None)
    main(smoke=not a.full, out_path=a.out, seeds=seeds)
