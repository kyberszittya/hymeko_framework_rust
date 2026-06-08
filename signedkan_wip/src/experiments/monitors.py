"""Monitor protocol + concrete implementations.

Observer pattern attached to Experiment.run() events. Each monitor
handles one concern (wall-clock, memory, AUC aggregation, JSONL
emission). New concerns = new monitor; do NOT add fields to the
runner.
"""
from __future__ import annotations

import json
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import CellSpec
    from .runner import ExperimentResult, SweepSummary


@runtime_checkable
class Monitor(Protocol):
    def on_cell_start(self, cell: "CellSpec") -> None: ...
    def on_cell_end(self, cell: "CellSpec",
                    result: "ExperimentResult") -> None: ...
    def on_sweep_end(self, summary: "SweepSummary") -> None: ...


# ── Concrete monitors ────────────────────────────────────────────────

class _NullMonitorMixin:
    """Default no-op implementations; subclass overrides what it needs."""
    def on_cell_start(self, cell: "CellSpec") -> None: pass
    def on_cell_end(self, cell: "CellSpec",
                    result: "ExperimentResult") -> None: pass
    def on_sweep_end(self, summary: "SweepSummary") -> None: pass


class WallMonitor(_NullMonitorMixin):
    """Tracks per-cell wall-clock; prints summary at sweep end."""
    def __init__(self) -> None:
        self._starts: dict[tuple, float] = {}
        self._walls: dict[tuple, float] = {}

    def on_cell_start(self, cell: "CellSpec") -> None:
        self._starts[(cell.dataset, cell.mode, cell.seed)] = time.time()

    def on_cell_end(self, cell, result) -> None:
        key = (cell.dataset, cell.mode, cell.seed)
        if key in self._starts:
            self._walls[key] = time.time() - self._starts[key]

    def on_sweep_end(self, summary) -> None:
        if not self._walls:
            return
        total = sum(self._walls.values())
        print(f"[WallMonitor] total={total:.1f}s n={len(self._walls)} "
              f"median={statistics.median(self._walls.values()):.1f}s")


class MemoryMonitor(_NullMonitorMixin):
    """Per-cell peak RSS via resource.getrusage (Linux: ru_maxrss in KB)."""
    def __init__(self) -> None:
        self._peaks: dict[tuple, float] = {}

    def on_cell_end(self, cell, result) -> None:
        # ru_maxrss is in KB on Linux; convert to MB.
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._peaks[(cell.dataset, cell.mode, cell.seed)] = peak_kb / 1024.0

    def on_sweep_end(self, summary) -> None:
        if not self._peaks:
            return
        peak = max(self._peaks.values())
        print(f"[MemoryMonitor] peak_rss={peak:.0f}MB across {len(self._peaks)} cells")


class AUCMonitor(_NullMonitorMixin):
    """Aggregates AUC by (dataset, mode); prints mean ± std at sweep end."""
    def __init__(self) -> None:
        self._aucs: dict[tuple[str, str], list[float]] = defaultdict(list)

    def on_cell_end(self, cell, result) -> None:
        if result.auc is not None:
            self._aucs[(cell.dataset, cell.mode)].append(result.auc)

    def on_sweep_end(self, summary) -> None:
        if not self._aucs:
            return
        print("[AUCMonitor] paired aggregates:")
        for (ds, mode), vals in sorted(self._aucs.items()):
            if not vals:
                continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {ds:<15} {mode:<8} n={len(vals)} auc={m:.4f} ± {s:.4f}")


class JSONLEmitter(_NullMonitorMixin):
    """Appends one JSON line per cell to ``path``.

    Path is resolved from ExperimentConfig.resolved_output_jsonl(); the
    runner injects it at construction.
    """
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_cell_end(self, cell, result) -> None:
        row = dict(result.raw_jsonl_row) if result.raw_jsonl_row else {}
        row.update({
            "_runner_dataset": cell.dataset,
            "_runner_mode": cell.mode,
            "_runner_seed": cell.seed,
            "_runner_wall_s": result.wall_s,
            "_runner_peak_rss_mb": result.peak_rss_mb,
        })
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


# ── Registry of monitor constructors ─────────────────────────────────

MONITOR_REGISTRY = {
    "WallMonitor":   lambda cfg: WallMonitor(),
    "MemoryMonitor": lambda cfg: MemoryMonitor(),
    "AUCMonitor":    lambda cfg: AUCMonitor(),
    "JSONLEmitter":  lambda cfg: JSONLEmitter(cfg.resolved_output_jsonl()),
}


def build_monitors(monitor_names, config) -> list[Monitor]:
    out: list[Monitor] = []
    for name in monitor_names:
        ctor = MONITOR_REGISTRY.get(name)
        if ctor is None:
            raise ValueError(f"unknown monitor '{name}'; known: "
                             f"{sorted(MONITOR_REGISTRY)}")
        out.append(ctor(config))
    return out
