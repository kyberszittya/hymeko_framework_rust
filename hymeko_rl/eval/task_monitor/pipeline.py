"""Live-pipeline tensor-contract guard — the runtime sibling of :class:`TensorContractMonitor`.

The formal monitor verifies *trajectories*; this verifies the *transition path itself*. A `PipelineSchemaLedger`
carries the canonical transition schema (ordered ``(field-name, dim)`` pairs derived from the env — obs / action /
reward / next_obs / done / privileged-z) and records the schema actually seen at each of the five pipeline stages:

    rollout → replay serialization → replay loading → critic training → evaluation

Recording a stage validates it against the canonical schema **immediately** and raises :class:`PipelineSchemaError`
the instant any field ordering or dimension drifts — so a mismatch between how a transition is written and how the
critic reads it aborts at the first offending stage rather than silently training a wrong-but-running critic (the
class of provenance / anchor-path bug the earlier RL smoke hit). ``verify_or_abort`` then confirms every stage was
actually exercised (no stage silently skipped). Full-transition stages (rollout/serialize/load) must match the
canonical schema exactly; consumer stages (critic/eval) must read a subset of the fields in canonical order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .consistency import TensorContractMonitor
from .submonitors import SubVerdict

STAGES = ("rollout", "replay_serialize", "replay_load", "critic_train", "eval")
# Stages that carry the WHOLE transition (must match the canonical schema exactly); the rest are consumers.
FULL_STAGES = frozenset({"rollout", "replay_serialize", "replay_load"})


class PipelineSchemaError(RuntimeError):
    """Raised when a pipeline stage's tensor schema or field ordering drifts from the canonical transition."""


@dataclass(frozen=True)
class TransitionField:
    name: str
    dim: int


def _fmt(fields: Sequence[TransitionField]) -> str:
    return "[" + ", ".join(f"{f.name}:{f.dim}" for f in fields) + "]"


@dataclass(frozen=True)
class TransitionSchema:
    """Canonical ordered ``(field-name, dim)`` layout of one stored transition, derived from the env."""

    fields: tuple[TransitionField, ...]

    @classmethod
    def from_env(cls, env: Any, action_dim: int, *, priv_enabled: bool) -> "TransitionSchema":
        obs_dim = int(np.prod(env.observation_space.shape))
        rows = [("obs", obs_dim), ("action", int(action_dim)), ("reward", 1),
                ("next_obs", obs_dim), ("done", 1)]
        if priv_enabled:
            pd = int(getattr(env, "privileged_dim", 0))
            rows += [("priv", pd), ("next_priv", pd)]
        return cls(tuple(TransitionField(n, d) for n, d in rows))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def hash(self) -> str:
        return TensorContractMonitor.field_hash(tuple(f"{f.name}:{f.dim}" for f in self.fields))


def flat_dim(x: Any, *, batched: bool) -> int:
    """Element width of a tensor/array: product of the trailing dims (dropping the batch dim iff ``batched``)."""
    shp = tuple(int(d) for d in np.asarray(x).shape) if not hasattr(x, "shape") else tuple(int(d) for d in x.shape)
    return int(np.prod(shp[1:])) if batched else int(np.prod(shp))


class PipelineSchemaLedger:
    """Records + validates the transition schema at each pipeline stage; aborts on the first drift."""

    def __init__(self, schema: TransitionSchema) -> None:
        self.schema = schema
        self.recorded: dict[str, tuple[TransitionField, ...]] = {}
        self.sealed = False

    def record(self, stage: str, fields: Sequence[tuple[str, int]]) -> None:
        """Record + immediately validate ``stage``'s ``(name, dim)`` fields against the canonical schema."""
        if stage not in STAGES:
            raise ValueError(f"unknown pipeline stage {stage!r}; expected one of {STAGES}")
        rec = tuple(TransitionField(str(n), int(d)) for n, d in fields)
        prev = self.recorded.get(stage)
        if prev is not None and prev != rec:
            raise PipelineSchemaError(f"{stage} schema changed mid-run: {_fmt(prev)} -> {_fmt(rec)}")
        self.recorded[stage] = rec
        self._validate_stage(stage, rec)

    def record_tensors(self, stage: str, named: Sequence[tuple[str, Any]], *, batched: bool) -> None:
        """Convenience: record a stage from live tensors, deriving each field's dim from its shape."""
        self.record(stage, [(n, flat_dim(t, batched=batched)) for n, t in named])

    def _validate_stage(self, stage: str, rec: tuple[TransitionField, ...]) -> None:
        canon = self.schema.fields
        cmap = {f.name: f.dim for f in canon}
        cnames = [f.name for f in canon]
        v: list[str] = []
        if stage in FULL_STAGES:
            if rec != canon:
                v.append(f"{stage}: {_fmt(rec)} != canonical {_fmt(canon)}")
        else:
            names = [f.name for f in rec]
            pos = [cnames.index(n) if n in cnames else -1 for n in names]
            if any(p < 0 for p in pos):
                v.append(f"{stage}: unknown field(s) {[n for n, p in zip(names, pos) if p < 0]}")
            elif pos != sorted(pos):
                v.append(f"{stage}: field order {names} violates canonical order {cnames}")
            for f in rec:
                if f.name in cmap and cmap[f.name] != f.dim:
                    v.append(f"{stage}: {f.name} dim {f.dim} != canonical {cmap[f.name]}")
        if v:
            raise PipelineSchemaError("; ".join(v))

    def verify(self) -> SubVerdict:
        """Completeness check: every stage was exercised (per-stage schema was validated at record time)."""
        missing = [s for s in STAGES if s not in self.recorded]
        violations = [f"stage never exercised: {s}" for s in missing]
        return SubVerdict("pipeline_schema", not violations, 1.0 if not violations else 0.0, violations, [], [])

    def verify_or_abort(self) -> SubVerdict:
        v = self.verify()
        if not v.passed:
            raise PipelineSchemaError("incomplete pipeline coverage: " + "; ".join(v.violations))
        self.sealed = True
        return v

    def stage_hashes(self) -> dict[str, str]:
        """Per-stage field-order hash (for logging / the report's tensor-contract line)."""
        return {s: TensorContractMonitor.field_hash(tuple(f"{f.name}:{f.dim}" for f in rec))
                for s, rec in self.recorded.items()}
