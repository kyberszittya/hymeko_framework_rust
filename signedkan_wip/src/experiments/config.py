"""Experiment configuration dataclasses + YAML loader.

Composes the existing ``runtime_config.py`` dataclasses
(TopKConfig, CycleCacheConfig, KernelConfig, ...) into a single
``ExperimentConfig`` that drives an Experiment run.

Loading: ``ExperimentConfig.from_yaml(Path("..."))``.

See: docs/architecture/experiments_runner/runner_architecture.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore  -- caller gets a clean error at from_yaml


# ── Atomic config dataclasses (mirror the YAML schema verbatim) ──────

@dataclass(frozen=True)
class CellSpec:
    """One (dataset, mode, seed) cell."""
    dataset: str
    mode: str       # "real" | "shuffle" | ...
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    family: str = "HSiKAN"
    hidden: int = 4
    n_epochs: int = 80
    mixed_tuples: tuple[str, ...] = ("c2", "c3", "c4", "c5", "w2", "w3")
    attention_m_e: str = "quaternion"
    attention_highway: bool = True
    attention_highway_kind: str = "edge_cr"
    max_k4: int = 200000


@dataclass(frozen=True)
class TopKConfig:
    """Local mirror of runtime_config.TopKConfig fields relevant at config-edit time."""
    mode: str = "per_vertex"   # per_vertex | per_vertex_adaptive | global
    k: int = 128
    pruner: str = "balance"
    scorer: str = "fraction_negative"
    m_v_min: int = 1
    m_v_max: int = 128
    m_v_c: float = 1.0


@dataclass(frozen=True)
class KernelConfig:
    triton: bool = False
    triton_backward: bool = False


@dataclass(frozen=True)
class CycleCacheConfig:
    enabled: bool = True
    dir: str = ".cache/hymeko_cycles"


@dataclass(frozen=True)
class OutputConfig:
    jsonl: str = "experiments/results/${name}.jsonl"
    summary: str = "reports/auto/${name}.md"


# ── ExperimentConfig (root) ──────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    """Root config; composes the atomic configs + sweep / monitors / env."""
    name: str
    description: str
    experiment_class: str   # registry key, e.g. "SweepExperiment"
    cells: tuple[CellSpec, ...]
    model: ModelConfig
    topk: TopKConfig
    kernel: KernelConfig
    cycle_cache: CycleCacheConfig
    output: OutputConfig
    env: dict[str, str] = field(default_factory=dict)
    monitor_names: tuple[str, ...] = ("WallMonitor", "MemoryMonitor",
                                       "AUCMonitor", "JSONLEmitter")

    # ── Loaders ────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        if yaml is None:
            raise RuntimeError("pyyaml not installed; pip install pyyaml")
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        cells = tuple(_expand_cells(d["cells"]))
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            experiment_class=d.get("experiment_class", "SweepExperiment"),
            cells=cells,
            model=ModelConfig(**(d.get("model", {}) or {})),
            topk=TopKConfig(**(d.get("topk", {}) or {})),
            kernel=KernelConfig(**(d.get("kernel", {}) or {})),
            cycle_cache=CycleCacheConfig(**(d.get("cycle_cache", {}) or {})),
            output=OutputConfig(**(d.get("output", {}) or {})),
            env={str(k): str(v) for k, v in (d.get("env", {}) or {}).items()},
            monitor_names=tuple(d.get("monitors",
                ("WallMonitor", "MemoryMonitor", "AUCMonitor", "JSONLEmitter"))),
        )

    # ── Helpers ────────────────────────────────────────────────────

    def resolved_output_jsonl(self) -> str:
        return self.output.jsonl.replace("${name}", self.name)

    def resolved_output_summary(self) -> str:
        return self.output.summary.replace("${name}", self.name)


def _expand_cells(spec_list: list[dict]) -> list[CellSpec]:
    """Expand the YAML cell list (each entry may carry a seeds list) into
    a flat list of CellSpec instances."""
    cells: list[CellSpec] = []
    for entry in spec_list:
        ds = entry["dataset"]
        mode = entry.get("mode", "real")
        seeds = entry.get("seeds", [entry.get("seed", 0)])
        for s in seeds:
            cells.append(CellSpec(dataset=ds, mode=mode, seed=int(s)))
    return cells
