"""Domain-neutral experiment specification for the canonical command layer.

One :class:`ExperimentSpec` describes *what to run* without committing to any domain's vocabulary (it is NOT RL-only —
``model_variant``/``execution_strategy`` are generic, not "policy"/"strategy"). Domain-specific knobs live in the opaque
``domain_options`` dict, which the matching domain adapter validates and consumes. The shared runner owns seeds,
resources, provenance and artifacts; adapters own only their domain semantics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    """A single reproducible unit of work, domain-agnostic.

    # Preconditions ``domain`` names a registered adapter; ``domain_options`` are that adapter's schema.
    # Invariants immutable once constructed; ``seed``/``repetition`` fully determine the run given the options.
    """
    domain: str                                  # "coin_delivery" | "cip" | "hypersignedlingam" | ...
    experiment_name: str
    model_variant: str = "default"               # a domain-named variant (NOT necessarily a "policy")
    execution_strategy: str = "default"          # a domain-named strategy (NOT necessarily an RL strategy)
    dataset_or_problem: str = ""                 # problem id / dataset ref, domain-interpreted
    seed: int = 0
    repetition: int = 0
    backend: str = "local"                       # "local" | "kato15" | "kato14" | ...
    resource_class: str = "cpu"                  # "cpu" | "gpu"
    artifact_root: str = "experiments/campaign"
    provenance: dict[str, Any] = field(default_factory=dict)
    domain_options: dict[str, Any] = field(default_factory=dict)

    def run_id(self) -> str:
        return (f"{self.domain}.{self.experiment_name}.{self.model_variant}.{self.execution_strategy}"
                f".{self.dataset_or_problem or 'na'}.s{self.seed}.r{self.repetition}").replace("/", "_")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentSpec":
        known = {f for f in cls.__dataclass_fields__}                # fail loud on typos, don't silently drop keys
        extra = set(d) - known
        if extra:
            raise ValueError(f"ExperimentSpec: unknown keys {sorted(extra)} (typo or wrong schema); known={sorted(known)}")
        return cls(**d)


@dataclass(frozen=True)
class CampaignSpec:
    """A matrix of runs sharing a name; ``runs`` is an explicit list of :class:`ExperimentSpec`."""
    name: str
    runs: list[ExperimentSpec]

    @classmethod
    def from_manifest(cls, path: str | Path) -> "CampaignSpec":
        m = json.loads(Path(path).read_text())
        if "runs" not in m or "name" not in m:
            raise ValueError(f"campaign manifest {path} must have 'name' and 'runs' (a list of specs)")
        defaults = m.get("defaults", {})
        runs = [ExperimentSpec.from_dict({**defaults, **r}) for r in m["runs"]]
        return cls(name=m["name"], runs=runs)
