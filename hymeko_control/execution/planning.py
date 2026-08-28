"""Generic planning contract (control-profile, scenario-agnostic).

A planning *role* (REACH, TRANSPORT, ...) is decoupled from the *implementation* that fulfils it (RRT-connect,
trajectory optimization, graph search, MPC, learned/amortized planners, ...). The general semantics NEVER encode
``REACH == RRT_CONNECT`` (addendum §4): that binding is a runtime registration a scenario supplies and a future
deployment can swap without touching the task-result contract, the CIP lifecycle, the GUI or the certificate (§13).

Command vs strategy (§5): a :class:`PlanningRequest` captures *what* is asked (role, start, goal, constraints); a
:class:`Planner` implementation captures *how* it is solved; a :class:`PlanningResult` returns the generic outcome
(status, plan, cost, timing, feasibility, provenance) with implementation-specific fields quarantined in
``diagnostics`` (§6) so the rest of the system never parses a planner's internals.

Torch-free / stdlib-only; frozen value types matching the ``hymeko_control.cip`` style.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .._frozen import freeze_mapping


class PlanningRole(str, Enum):
    """The semantic role a plan fills within a task. Open vocabulary — a scenario may bind any role to any planner."""

    REACH = "reach"
    TRANSPORT = "transport"
    APPROACH = "approach"
    RETREAT = "retreat"


class PlanningStatus(str, Enum):
    """Outcome status of a planning attempt, independent of which planner produced it."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"  # e.g. the plan is executed inside a deployed pipeline and feasibility is known post-hoc


@dataclass(frozen=True)
class PlanningRequest:
    """*What* is requested of a planner (§5). Embodiment-agnostic; scenario detail goes in ``constraints``."""

    role: PlanningRole
    start: tuple[float, ...] = ()
    goal: tuple[float, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", tuple(float(x) for x in self.start))
        object.__setattr__(self, "goal", tuple(float(x) for x in self.goal))
        object.__setattr__(self, "constraints", freeze_mapping(self.constraints))


@dataclass(frozen=True)
class PlanningResult:
    """The generic outcome of a planning attempt. ``diagnostics`` holds implementation-specific fields (e.g. RRT node
    count, shortcut ratio) that generic consumers must NOT parse; ``provenance`` names the owner and postprocess so
    scientific provenance stays explicit even under a general abstraction (§11)."""

    role: PlanningRole
    planner: str
    status: PlanningStatus
    feasible: bool = False
    plan: tuple[Any, ...] = ()
    cost: float | None = None
    timing: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", tuple(self.plan))
        object.__setattr__(self, "timing", freeze_mapping(self.timing))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))


@runtime_checkable
class Planner(Protocol):
    """A planner implementation bound to a role (§4). Interchangeable: any object exposing ``role``, ``name`` and a
    ``plan`` map from request to result satisfies the contract. A scenario registers one per role; swapping it is the
    ONLY change a planner replacement requires (§13)."""

    role: PlanningRole
    name: str

    def plan(self, request: PlanningRequest) -> PlanningResult: ...


class PlannerRegistry:
    """role -> bound planner implementation. The registry is the single seam a future planner swap goes through: the
    task result, CIP lifecycle, GUI and certificate never name a concrete planner — they ask the registry (§7, §13).

    Not a global (§6.5 #11): a scenario constructs and threads its own registry explicitly.
    """

    def __init__(self) -> None:
        self._by_role: dict[PlanningRole, Planner] = {}

    def register(self, planner: Planner) -> PlannerRegistry:
        """Bind ``planner`` to its declared role, replacing any prior binding. Returns self for chaining."""
        if not isinstance(planner.role, PlanningRole):
            raise TypeError(f"planner.role must be a PlanningRole, got {type(planner.role)!r}")
        self._by_role[planner.role] = planner
        return self

    def resolve(self, role: PlanningRole) -> Planner:
        """Return the planner currently bound to ``role``.

        # Errors
        ``KeyError`` if no planner is registered for the role — a task must not silently proceed with no planner.
        """
        if role not in self._by_role:
            raise KeyError(f"no planner registered for role {role.value!r}")
        return self._by_role[role]

    def binding(self, role: PlanningRole) -> str:
        """The *name* of the planner currently bound to ``role`` (for provenance/event labelling)."""
        return self.resolve(role).name

    def roles(self) -> tuple[PlanningRole, ...]:
        return tuple(self._by_role)
