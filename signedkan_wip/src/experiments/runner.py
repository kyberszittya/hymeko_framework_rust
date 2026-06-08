"""Experiment runner classes.

``Experiment`` ABC + three concrete subclasses:
  * ``SingleCellExperiment`` -- one (dataset, mode, seed) cell.
  * ``SweepExperiment``      -- a grid of cells, sequential.
  * ``KomondorArrayExperiment`` -- emit sbatch invocation; do NOT
                                   re-implement SLURM.

The single-cell runner wraps ``run_final_cell.main()`` for backward
compatibility -- the 1295-LOC training body is NOT rewritten in this
commit; only orchestrated.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Type

from .config import CellSpec, ExperimentConfig
from .monitors import Monitor, build_monitors
from .registry import ExperimentRegistry


# ── Result types ─────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """One cell's outcome."""
    auc: float | None = None
    f1m: float | None = None
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    n_params: int = 0
    config_echo: dict[str, Any] = field(default_factory=dict)
    raw_jsonl_row: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class SweepSummary:
    """Aggregated outcome over a SweepExperiment."""
    per_cell: list[tuple[CellSpec, ExperimentResult]]
    total_wall_s: float
    n_cells: int
    n_failed: int


# ── Experiment ABC ───────────────────────────────────────────────────

class Experiment(ABC):
    """Abstract base; subclasses register themselves via __init_subclass__."""
    def __init__(self) -> None:
        self.monitors: list[Monitor] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register concrete subclasses by their class name.
        ExperimentRegistry.register(cls.__name__, cls)

    @abstractmethod
    def run(self, config: ExperimentConfig) -> Any:
        ...

    def add_monitor(self, m: Monitor) -> None:
        self.monitors.append(m)

    def _notify_cell_start(self, cell: CellSpec) -> None:
        for m in self.monitors:
            m.on_cell_start(cell)

    def _notify_cell_end(self, cell: CellSpec, result: ExperimentResult) -> None:
        for m in self.monitors:
            m.on_cell_end(cell, result)

    def _notify_sweep_end(self, summary: SweepSummary) -> None:
        for m in self.monitors:
            m.on_sweep_end(summary)


# ── SingleCellExperiment ─────────────────────────────────────────────

class SingleCellExperiment(Experiment):
    """One (dataset, mode, seed) cell -> ExperimentResult.

    Wraps ``run_final_cell.main()`` by invoking it as a subprocess so
    the existing argparse + environment-var contract is preserved.
    No numerical behaviour change vs the historical shell scripts.
    """

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        assert len(config.cells) == 1, \
            f"SingleCellExperiment requires exactly 1 cell; got {len(config.cells)}"
        cell = config.cells[0]
        return self._run_cell(cell, config)

    # Internal: also used by SweepExperiment.
    def _run_cell(self, cell: CellSpec,
                  config: ExperimentConfig) -> ExperimentResult:
        self._notify_cell_start(cell)
        env = _compose_env(cell, config)
        cmd = _compose_cmd(cell, config)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        wall_s = time.time() - t0
        result = _parse_subprocess_output(proc, wall_s)
        self._notify_cell_end(cell, result)
        return result


# ── SweepExperiment ──────────────────────────────────────────────────

class SweepExperiment(Experiment):
    """Sequential sweep over config.cells; delegates per-cell to SingleCell."""

    def run(self, config: ExperimentConfig) -> SweepSummary:
        delegate = SingleCellExperiment()
        # Sweep monitors are attached to the SWEEP (this object); we
        # don't re-attach to the delegate so events aren't duplicated.
        # Instead the delegate runs without monitors and we notify
        # at the sweep level using its return.
        per_cell: list[tuple[CellSpec, ExperimentResult]] = []
        t0 = time.time()
        n_failed = 0
        for cell in config.cells:
            self._notify_cell_start(cell)
            r = delegate._run_cell_silent(cell, config)
            self._notify_cell_end(cell, r)
            per_cell.append((cell, r))
            if r.error is not None or r.auc is None:
                n_failed += 1
        summary = SweepSummary(
            per_cell=per_cell,
            total_wall_s=time.time() - t0,
            n_cells=len(config.cells),
            n_failed=n_failed,
        )
        self._notify_sweep_end(summary)
        return summary


# Add a silent variant of _run_cell to avoid double-notification when
# called via SweepExperiment.
def _silent_run_cell(self: SingleCellExperiment, cell: CellSpec,
                     config: ExperimentConfig) -> ExperimentResult:
    env = _compose_env(cell, config)
    cmd = _compose_cmd(cell, config)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_s = time.time() - t0
    return _parse_subprocess_output(proc, wall_s)

SingleCellExperiment._run_cell_silent = _silent_run_cell  # type: ignore[attr-defined]


# ── KomondorArrayExperiment ─────────────────────────────────────────

class KomondorArrayExperiment(Experiment):
    """Emit an sbatch invocation against the canonical Komondor
    submitter. Does NOT submit (user-gated)."""

    def run(self, config: ExperimentConfig) -> list[str]:
        # Returns the sbatch argv as a list of strings, for the user
        # to inspect / submit manually.
        submitter = "docs/komondor_setup/submit_hsikan_edge_cr_array.sh"
        cells_arg = ",".join(
            f"{c.dataset}:{c.mode}:cold" for c in config.cells
        )
        # Defer per-class --time to scripts/estimate_slurm_time.py
        # (already lives separately; KomondorArrayExperiment is just
        # a shell-command generator, not a re-implementation).
        argv = [
            "bash", submitter, "full",
            f"AUDIT_K_TAG={config.name}",
        ]
        print(f"[KomondorArrayExperiment] sbatch invocation (not executed):")
        print("  " + " ".join(argv))
        print(f"  cells: {cells_arg}")
        return argv


# ── Shared helpers ───────────────────────────────────────────────────

def _compose_env(cell: CellSpec, config: ExperimentConfig) -> dict[str, str]:
    """Build the subprocess env from the config + cell."""
    env = dict(os.environ)
    # Model + tuple config -> HSIKAN_* env vars (matches the historical contract)
    env["HSIKAN_MIXED_TUPLES"] = ",".join(config.model.mixed_tuples)
    env["HSIKAN_ATTENTION_M_E"] = config.model.attention_m_e
    env["HSIKAN_ATTENTION_HIGHWAY"] = "1" if config.model.attention_highway else "0"
    env["HSIKAN_ATTENTION_HIGHWAY_KIND"] = config.model.attention_highway_kind
    env["HSIKAN_CYCLE_BATCH"] = "2000"
    env["HSIKAN_MAX_K2"] = "200000"
    env["HSIKAN_MAX_K3"] = "200000"
    # TopK
    env["HSIKAN_TOPK_MODE"] = config.topk.mode
    env["HSIKAN_TOPK_K"] = str(config.topk.k)
    env["HSIKAN_TOPK_PRUNER"] = config.topk.pruner
    env["HSIKAN_TOPK_SCORER"] = config.topk.scorer
    if config.topk.mode == "per_vertex_adaptive":
        env["HSIKAN_TOPK_M_V_MIN"] = str(config.topk.m_v_min)
        env["HSIKAN_TOPK_M_V_MAX"] = str(config.topk.m_v_max)
        env["HSIKAN_TOPK_M_V_C"] = str(config.topk.m_v_c)
    # Triton kernel
    if config.kernel.triton:
        env["HSIKAN_TRITON_KERNEL"] = "1"
    if config.kernel.triton_backward:
        env["HSIKAN_TRITON_BACKWARD"] = "1"
    # Cycle cache
    if config.cycle_cache.enabled:
        env["HYMEKO_CYCLE_CACHE"] = "1"
        env["HYMEKO_CYCLE_CACHE_DIR"] = config.cycle_cache.dir
    # User-provided env overrides (PYTORCH_CUDA_ALLOC_CONF etc.)
    env.update(config.env)
    # PYTHONPATH: ensure repo root is on it so signedkan_wip is importable.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." + (":" + pp if pp else "")
    return env


def _compose_cmd(cell: CellSpec, config: ExperimentConfig) -> list[str]:
    """Build the python invocation argv for one cell."""
    cmd = [
        sys.executable, "-m", "signedkan_wip.experiments.runs.run_final_cell",
        "--dataset", cell.dataset,
        "--hidden", str(config.model.hidden),
        "--n-epochs", str(config.model.n_epochs),
        "--max-k4", str(config.model.max_k4),
        "--seed", str(cell.seed),
    ]
    if cell.mode == "shuffle":
        cmd.append("--shuffle-train-signs")
    return cmd


def _parse_subprocess_output(proc: subprocess.CompletedProcess,
                             wall_s: float) -> ExperimentResult:
    """Find the JSON result line in stdout; build ExperimentResult."""
    import json
    if proc.returncode != 0:
        return ExperimentResult(
            wall_s=wall_s,
            error=f"rc={proc.returncode}; stderr tail: "
                  f"{proc.stderr[-500:] if proc.stderr else ''}",
        )
    json_line = None
    for line in (proc.stdout or "").splitlines():
        s = line.strip()
        if s.startswith('{"dataset"'):
            json_line = s   # take the LAST matching line
    if json_line is None:
        return ExperimentResult(wall_s=wall_s, error="no JSON result line in stdout")
    try:
        row = json.loads(json_line)
    except json.JSONDecodeError as e:
        return ExperimentResult(wall_s=wall_s, error=f"JSON parse: {e}")
    return ExperimentResult(
        auc=row.get("auc"),
        f1m=row.get("f1m"),
        wall_s=wall_s,
        n_params=int(row.get("n_params", 0)),
        config_echo=row,
        raw_jsonl_row=row,
    )


# ── Convenience: load + attach monitors + run ────────────────────────

def run_from_config(config: ExperimentConfig) -> Any:
    """Top-level entry. Picks the right Experiment subclass from the
    registry, attaches the configured monitors, runs."""
    klass = ExperimentRegistry.lookup(config.experiment_class)
    exp = klass()
    for m in build_monitors(config.monitor_names, config):
        exp.add_monitor(m)
    return exp.run(config)
