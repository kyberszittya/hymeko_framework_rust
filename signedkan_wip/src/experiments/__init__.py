"""Experiment runner framework.

One unified entry point for HSiKAN / Gömb / SGCN / SGT experiments;
replaces the historical ~221 launcher scripts in
``signedkan_wip/experiments/run_*.{sh,py}``.

See:
  - ``docs/architecture/experiments_runner/runner_architecture.md``
  - CLAUDE.md §6.5 #3 (the rule this implements)

Public API:
  from signedkan_wip.src.experiments import (
      Experiment, SingleCellExperiment, SweepExperiment,
      ExperimentConfig, ExperimentResult, SweepSummary,
      Monitor, WallMonitor, MemoryMonitor, AUCMonitor, JSONLEmitter,
      ExperimentRegistry,
  )
"""
from .config import (
    CellSpec, CycleCacheConfig, ExperimentConfig, KernelConfig,
    ModelConfig, OutputConfig, TopKConfig,
)
from .monitors import (
    AUCMonitor, JSONLEmitter, MemoryMonitor, Monitor, WallMonitor,
)
from .registry import ExperimentRegistry
from .runner import (
    Experiment, ExperimentResult, SingleCellExperiment, SweepExperiment,
    SweepSummary,
)

__all__ = [
    "AUCMonitor", "CellSpec", "CycleCacheConfig", "Experiment",
    "ExperimentConfig", "ExperimentRegistry", "ExperimentResult",
    "JSONLEmitter", "KernelConfig", "MemoryMonitor", "ModelConfig",
    "Monitor", "OutputConfig", "SingleCellExperiment", "SweepExperiment",
    "SweepSummary", "TopKConfig", "WallMonitor",
]
