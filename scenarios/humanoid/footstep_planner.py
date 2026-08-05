r"""Footstep planning over the SHARED Rust A\* engine (``akoire::astar`` via ``hymeko.astar_plan``).

The Rust ``SearchProblem`` / ``astar`` planner framework — the *same* search HOTARU's HIVE-delta
planner uses — bound into Python: the humanoid footstep planner defines its foothold graph
(successors, goal, heuristic) as Python callables over integer foothold ids, and the **search loop
runs in Rust** (``akoire::astar``). This is the real code-sharing "mirror": one engine drives both
structure-synthesis (HOTARU) and motion (footsteps).

Backend ``auto`` prefers ``hymeko.astar_plan`` when the built extension exposes it, else falls back to
the pure-Python reference A\* (``hymeko_rl.control.graph_planner.astar``) — the same algorithm, so the
plan is identical in cost. The planner therefore runs everywhere while using the shared Rust engine
wherever it is available.

# Preconditions: a ``SteppingField`` with passable start/goal. # Postconditions: ``plan_footsteps``
#   returns a sequence of adjacent, passable footholds ending at the goal, or ``None`` if unreachable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from math import sqrt

Cell = "tuple[int, int]"
_DIAG = sqrt(2.0)


class Backend(str, Enum):
    r"""Which A\* implementation runs the search."""

    AUTO = "auto"      # rust if hymeko.astar_plan is present, else python
    RUST = "rust"      # the shared akoire::astar via hymeko.astar_plan (fails if unavailable)
    PYTHON = "python"  # the pure-Python reference A* (hymeko_rl.control.graph_planner)


def _rust_astar_plan() -> "Callable | None":
    """The bound Rust planner (``hymeko.astar_plan``) if the built extension exposes it, else ``None``."""
    try:
        import hymeko
    except ImportError:
        return None
    return getattr(hymeko, "astar_plan", None)


@dataclass(frozen=True)
class SteppingField:
    r"""A discrete stepping-stone field: cells are candidate footholds; ``blocked`` are unavailable.

    A foothold cell ``(i, j)`` maps to the single ``i64`` node ``j * width + i`` the shared Rust A\*
    searches over, so no per-edge Python object crosses the boundary — the id sequence *is* the plan.

    # Preconditions: ``width, height >= 1``; ``connectivity in {4, 8}``.
    # Invariants: ``cell_id`` and ``id_cell`` are mutual inverses on in-bounds cells.
    """

    width: int
    height: int
    blocked: frozenset  # of Cell
    connectivity: int = 4

    def __post_init__(self) -> None:
        assert self.width >= 1 and self.height >= 1, "field must be non-empty"
        assert self.connectivity in (4, 8), "connectivity must be 4 or 8"

    def cell_id(self, cell: "tuple[int, int]") -> int:
        return cell[1] * self.width + cell[0]

    def id_cell(self, node: int) -> "tuple[int, int]":
        return (node % self.width, node // self.width)

    def in_bounds(self, cell: "tuple[int, int]") -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def passable(self, cell: "tuple[int, int]") -> bool:
        return self.in_bounds(cell) and cell not in self.blocked

    def _steps(self) -> "list[tuple[int, int]]":
        base = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            base += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return base

    def neighbours_id(self, node: int) -> "list[tuple[int, float]]":
        """Passable successors of ``node`` as ``(next_id, step_cost)`` (diagonal steps cost ``sqrt(2)``)."""
        i, j = self.id_cell(node)
        out: "list[tuple[int, float]]" = []
        for di, dj in self._steps():
            c = (i + di, j + dj)
            if self.passable(c):
                out.append((self.cell_id(c), 1.0 if (di == 0 or dj == 0) else _DIAG))
        return out

    def heuristic_id(self, node: int, goal: "tuple[int, int]") -> float:
        """Admissible remaining-cost estimate to ``goal``: Manhattan (4-conn) or octile (8-conn)."""
        i, j = self.id_cell(node)
        dx, dy = abs(goal[0] - i), abs(goal[1] - j)
        if self.connectivity == 8:
            return float((dx + dy) + (_DIAG - 2.0) * min(dx, dy))  # octile, admissible & tight
        return float(dx + dy)                                      # Manhattan


def _select_backend(backend: Backend) -> Backend:
    """Resolve ``AUTO`` to ``RUST``/``PYTHON`` by availability; validate an explicit ``RUST`` request."""
    if backend is Backend.PYTHON:
        return Backend.PYTHON
    available = _rust_astar_plan() is not None
    if backend is Backend.RUST and not available:
        raise RuntimeError("backend='rust' requested but hymeko.astar_plan is unavailable")
    return Backend.RUST if available else Backend.PYTHON


def _reference_astar() -> "Callable":
    r"""The pure-Python reference A\* (``hymeko_rl.control.graph_planner.astar``).

    Loaded directly from its file rather than via ``import hymeko_rl.control.graph_planner`` because
    ``hymeko_rl/__init__.py`` eagerly imports the torch-backed policy stack; the reference module
    itself is stdlib + numpy only, so a by-path load keeps the fallback lightweight (and torch-free).
    It is the *same* code — no re-implementation (§6.1).
    """
    import importlib.util
    import sys
    from pathlib import Path

    name = "hymeko_rl_graph_planner_ref"
    if name in sys.modules:                                       # already loaded once this process
        return sys.modules[name].astar
    gp = Path(__file__).resolve().parents[2] / "hymeko_rl" / "control" / "graph_planner.py"
    spec = importlib.util.spec_from_file_location(name, gp)
    if spec is None or spec.loader is None:                       # pragma: no cover - defensive
        raise ImportError(f"cannot load the reference A* from {gp}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so dataclasses can resolve `__module__`
    spec.loader.exec_module(mod)
    return mod.astar


def _python_plan(start_id: int, neighbours: "Callable[[int], Iterable[tuple[int, float]]]",
                 is_goal: "Callable[[int], bool]", heuristic: "Callable[[int], float]",
                 max_expansions: int) -> "list[int] | None":
    """Fallback: run the pure-Python reference A\\*; return the node path minus the start."""
    node_path = _reference_astar()(start_id, is_goal, lambda n: iter(neighbours(n)), heuristic,
                                   max_expansions=max_expansions)
    return None if node_path is None else node_path[1:]  # drop start ⇒ the stepped-into footholds


def plan_footsteps(field: SteppingField, start: "tuple[int, int]", goal: "tuple[int, int]", *,
                   backend: Backend = Backend.AUTO,
                   max_expansions: int = 200_000) -> "list[tuple[int, int]] | None":
    r"""Plan a foothold sequence from ``start`` to ``goal`` over ``field`` using the shared A\* engine.

    Returns the footholds from the first step to the goal (``start`` excluded), or ``None`` if the goal
    is unreachable within ``max_expansions``.

    # Preconditions: ``start`` and ``goal`` are passable. # Postconditions: every returned cell is
    #   passable and adjacent to its predecessor; the last cell is ``goal`` (empty list if start == goal).
    """
    assert field.passable(start), "start must be a passable foothold"
    assert field.passable(goal), "goal must be a passable foothold"
    goal_id = field.cell_id(goal)

    def neighbours(node: int) -> "list[tuple[int, float]]":
        return field.neighbours_id(node)

    def is_goal(node: int) -> bool:
        return node == goal_id

    def heuristic(node: int) -> float:
        return field.heuristic_id(node, goal)

    if _select_backend(backend) is Backend.RUST:
        rust_plan = _rust_astar_plan()
        path_ids = rust_plan(field.cell_id(start), neighbours, is_goal, heuristic, max_expansions)
    else:
        path_ids = _python_plan(field.cell_id(start), neighbours, is_goal, heuristic, max_expansions)

    return None if path_ids is None else [field.id_cell(n) for n in path_ids]
