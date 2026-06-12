"""Load committed accuracy/param records for the slide-18 accuracy-vs-cost view.

Pure (no torch): reads the committed 5-seed JSONs and returns per-method
``AccuracyRecord``s keyed by ``(dataset, method)``. Two sources with **different
fields and provenance**, kept explicit so the Pareto never launders one as the
other:

- ``phase8_bitcoin_5seed.json`` — apples-to-apples panel, **untuned** baselines
  + HSiKAN-leanest; AUC field ``test_auc``.
- ``bitcoin_optuna_best_5seed_2026_05_13.jsonl`` — the **tuned** HSiKAN point;
  AUC field ``auc``.

The cost axis on the Pareto is ``n_params`` (config-intrinsic, present in every
record), so no latency is read here and no cross-host mixing is possible.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# phase8 ``arch`` → display method name. Only these are surfaced; an arch not in
# the map is ignored (not invented).
PHASE8_METHODS = {
    "sgcn_balance": "SGCN",
    "sigat_attn": "SiGAT",
    "mlp_blind": "MLP-blind",
    "gcn_blind": "GCN-blind",
    "hsikan_mixed_leanest": "HSiKAN-leanest",
    "signedkan_L1": "SignedKAN-L1",
}


@dataclass(frozen=True)
class AccuracyRecord:
    """One method's committed accuracy on one dataset.

    Invariants: ``0 <= mean_auc <= 1``; ``n_seeds >= 1``; ``n_params > 0``.
    ``tuned`` and ``source`` carry the provenance the Pareto caption needs.
    """
    dataset: str
    method: str
    mean_auc: float
    std_auc: float
    n_params: int
    n_seeds: int
    tuned: bool
    source: str


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _group_records(
    runs: list[dict[str, Any]], auc_field: str, method_of: Any,
    tuned: bool, source: str,
) -> dict[tuple[str, str], AccuracyRecord]:
    """Group raw run dicts into per-(dataset, method) AccuracyRecords.

    ``method_of(run) -> str | None`` maps a run to its display method (or None
    to drop it). Runs with a null/missing AUC are skipped.
    """
    buckets: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for r in runs:
        method = method_of(r)
        if method is None:
            continue
        auc = r.get(auc_field)
        if auc is None:
            continue
        key = (r["dataset"], method)
        slot = buckets.setdefault(key, {"auc": [], "params": []})
        slot["auc"].append(float(auc))
        if r.get("n_params") is not None:
            slot["params"].append(int(r["n_params"]))
    out: dict[tuple[str, str], AccuracyRecord] = {}
    for (dataset, method), slot in buckets.items():
        if not slot["auc"]:
            continue
        mean, std = _mean_std(slot["auc"])
        params = slot["params"][0] if slot["params"] else 0
        out[(dataset, method)] = AccuracyRecord(
            dataset=dataset, method=method, mean_auc=mean, std_auc=std,
            n_params=params, n_seeds=len(slot["auc"]), tuned=tuned, source=source,
        )
    return out


def load_phase8_accuracy(
    path: Any, methods: dict[str, str] | None = None,
) -> dict[tuple[str, str], AccuracyRecord]:
    """Per-method untuned 5-seed AUC from the apples-to-apples phase8 panel.

    Preconditions: ``path`` is a phase8 json with a ``runs`` list.
    Postconditions: keys are ``(dataset, display_method)``; only ``methods``
    archs appear; missing arch ⇒ absent key (never a fabricated record).
    """
    methods = methods or PHASE8_METHODS
    path = Path(path)
    runs = json.loads(path.read_text())["runs"]
    return _group_records(
        runs, auc_field="test_auc",
        method_of=lambda r: methods.get(r.get("arch")),
        tuned=False, source=path.name,
    )


def load_optuna_best(
    path: Any, method: str = "HSiKAN-optuna",
) -> dict[tuple[str, str], AccuracyRecord]:
    """The tuned HSiKAN point(s) from the optuna 5-seed JSONL (field ``auc``)."""
    path = Path(path)
    runs = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    return _group_records(
        runs, auc_field="auc", method_of=lambda r: method,
        tuned=True, source=path.name,
    )


def merge_records(
    *record_maps: dict[tuple[str, str], AccuracyRecord],
) -> dict[tuple[str, str], AccuracyRecord]:
    """Union of accuracy maps; later maps win on key collision."""
    merged: dict[tuple[str, str], AccuracyRecord] = {}
    for m in record_maps:
        merged.update(m)
    return merged


__all__ = [
    "AccuracyRecord", "PHASE8_METHODS",
    "load_phase8_accuracy", "load_optuna_best", "merge_records",
]
