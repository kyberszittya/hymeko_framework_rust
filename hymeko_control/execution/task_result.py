"""Generic task-execution result envelope (control-profile, scenario-agnostic).

This is the ONE canonical, embodiment-independent contract for "what happened when a task ran". Every scenario adapter
(coin delivery, pick-and-place, AIBO, humanoid) returns a :class:`TaskResult` — or a *specialization* of it — and every
generic consumer (the GUI execution framework, loggers, verifiers) depends ONLY on the fields declared here, never on a
scenario's extra fields. Coin-specific quantities (selected yaw, brake zone, dtz) live in a :class:`TaskResult`
*subclass*; the generic surface does not know they exist.

Design (mirrors ``hymeko_control.cip``): frozen dataclasses with ``freeze_mapping`` for maps, ``str``-enums for closed
vocabularies, ``provenance`` mappings for auditability. Torch-free / stdlib-only. The ``certificate`` field REUSES the
existing :class:`hymeko_control.cip.certificate.CertificateResult` — no parallel certificate abstraction (addendum §1).

Timestamps (``t``) are MONOTONIC LOGICAL indices, not physical time (§16: never conflate event rate with physical
time). A sample's physical time, if any, belongs in its ``payload``/``provenance``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .._frozen import freeze_mapping
from ..cip.certificate import CertificateResult


class TaskStatus(str, Enum):
    """Lifecycle status of one task execution. Generic across embodiments."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"  # safety abort / precondition not met (e.g. no feasible plan)


class TaskEventKind(str, Enum):
    """Generic semantic event vocabulary (§8). Implementation-specific event *names* (e.g. coin "BrakeSelected") are
    carried in :attr:`TaskEvent.label`; the *kind* stays embodiment-independent so a generic timeline renderer works for
    every scenario. ``PLAN_REQUESTED`` / ``PLAN_COMPUTED`` payloads carry the planning ``role`` and bound ``planner``."""

    TASK_STARTED = "task_started"
    PLAN_REQUESTED = "plan_requested"
    PLAN_COMPUTED = "plan_computed"
    OPTION_SELECTED = "option_selected"
    PHASE_CHANGED = "phase_changed"
    EXECUTION_PROGRESS = "execution_progress"
    CERTIFICATE_UPDATED = "certificate_updated"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"


@dataclass(frozen=True)
class TaskEvent:
    """One semantic event on a task's timeline.

    # Invariants
    Frozen; ``payload`` is a read-only view. ``t`` is a monotonic logical index (not physical time). ``label`` is an
    optional implementation-specific event name (e.g. ``"ReachPlanned"``) that refines :attr:`kind` for display without
    changing the generic vocabulary a generic consumer switches on.
    """

    t: int
    kind: TaskEventKind
    label: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_mapping(self.payload))


@dataclass(frozen=True)
class TrajectorySample:
    """One observational sample of an execution trajectory (§9).

    Generic envelope: ``t`` (monotonic logical index), ``phase`` (the control phase active at the sample), a generic
    ``payload`` (embodiment-specific projections — coin puts ``coin_x``/``coin_y``/``dtz_mm``/``speed`` here; a
    manipulator would put an ee-pose), and ``provenance`` (how the sample was obtained — e.g. read-only frame hook).

    # Invariants
    Frozen; observational only — constructing or reading a sample never mutates physics (§13).
    """

    t: int
    phase: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))


@dataclass(frozen=True)
class TaskResult:
    """The generic result of executing one task. Specializations extend this; generic consumers depend ONLY on it.

    Covers (addendum §2): task identity (:attr:`task_id`), status + success/failure (:attr:`status`,
    :attr:`succeeded`), failure classification (:attr:`failure_class`), certificate (:attr:`certificate`, reusing the
    CIP-0 contract), metrics, timing, provenance, events and trajectory/artifacts.

    # Invariants
    Frozen; all mappings are read-only views; ``events`` / ``trajectory`` are tuples. ``succeeded`` is redundant with
    ``status == SUCCEEDED`` and MUST agree — a post-init check enforces it, so a specialization cannot desync them.
    """

    task_id: str
    status: TaskStatus
    succeeded: bool
    certificate: CertificateResult | None = None
    failure_class: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[TaskEvent, ...] = ()
    trajectory: tuple[TrajectorySample, ...] = ()

    def __post_init__(self) -> None:
        if self.succeeded != (self.status is TaskStatus.SUCCEEDED):
            raise ValueError(
                f"TaskResult: succeeded={self.succeeded} disagrees with status={self.status.value}"
            )
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))
        object.__setattr__(self, "timing", freeze_mapping(self.timing))
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))

    def events_of(self, kind: TaskEventKind) -> tuple[TaskEvent, ...]:
        """All events of a given generic kind, in order (generic consumers filter by kind, never by label)."""
        return tuple(e for e in self.events if e.kind is kind)

    def timestamps_monotonic(self) -> bool:
        """True iff event and trajectory logical timestamps are non-decreasing (§16 guard: monotonic, not physical)."""
        ev = [e.t for e in self.events]
        tr = [s.t for s in self.trajectory]
        return ev == sorted(ev) and tr == sorted(tr)
