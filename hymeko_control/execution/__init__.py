"""Generic task-execution contracts (scenario-agnostic, torch-free).

This sub-package is the embodiment-independent layer ABOVE the CIP-0 value types: it declares the ONE canonical
task-result envelope every scenario returns, the generic semantic event / trajectory model a generic GUI renders, and
the role-based planning contract that keeps planner *implementations* interchangeable (REACH is a role, RRT-connect is
one implementation — never hardcoded). A scenario adapter specializes :class:`TaskResult` and registers its planners;
generic consumers depend only on the contracts here.
"""
from __future__ import annotations

from .planning import (
    Planner,
    PlannerRegistry,
    PlanningRequest,
    PlanningResult,
    PlanningRole,
    PlanningStatus,
)
from .task_result import (
    TaskEvent,
    TaskEventKind,
    TaskResult,
    TaskStatus,
    TrajectorySample,
)

__all__ = [
    "Planner",
    "PlannerRegistry",
    "PlanningRequest",
    "PlanningResult",
    "PlanningRole",
    "PlanningStatus",
    "TaskEvent",
    "TaskEventKind",
    "TaskResult",
    "TaskStatus",
    "TrajectorySample",
]
