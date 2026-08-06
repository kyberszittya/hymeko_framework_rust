"""Footstep planner over the shared Rust A* — python-backend correctness + rust-backend parity.

The python-backend tests run everywhere (they use the pure-Python reference A*). The rust-backend
tests exercise `hymeko.astar_plan` (the bound `akoire::astar`) and skip until the extension is rebuilt
with the binding — so the suite is green before and after `maturin develop`.
"""

from __future__ import annotations

import pytest

from scenarios.humanoid.footstep_planner import (
    Backend,
    SteppingField,
    _rust_astar_plan,
    plan_footsteps,
)

_HAS_RUST = _rust_astar_plan() is not None
_skip_rust = pytest.mark.skipif(not _HAS_RUST, reason="hymeko.astar_plan not built (rebuild the extension)")


def _wall_field() -> SteppingField:
    """4×4 field with a two-cell wall at column i=1 (rows j=0,1); the gap is at j≥2."""
    return SteppingField(width=4, height=4, blocked=frozenset({(1, 0), (1, 1)}))


def test_python_backend_detours_the_wall() -> None:
    field = _wall_field()
    path = plan_footsteps(field, (0, 0), (3, 0), backend=Backend.PYTHON, max_expansions=1000)
    assert path is not None
    assert path[-1] == (3, 0)                                   # ends at the goal
    assert all(field.passable(c) for c in path)                # never steps into the wall
    assert (1, 0) not in path and (1, 1) not in path
    assert len(path) == 7                                       # Manhattan 3 + a +4 detour over the wall


def test_python_backend_unreachable_returns_none() -> None:
    field = SteppingField(width=3, height=1, blocked=frozenset({(1, 0)}))  # wall severs the row
    assert plan_footsteps(field, (0, 0), (2, 0), backend=Backend.PYTHON) is None


def test_start_equals_goal_is_empty_plan() -> None:
    field = _wall_field()
    assert plan_footsteps(field, (0, 0), (0, 0), backend=Backend.PYTHON) == []


def test_diagonal_field_uses_octile() -> None:
    field = SteppingField(width=3, height=3, blocked=frozenset(), connectivity=8)
    path = plan_footsteps(field, (0, 0), (2, 2), backend=Backend.PYTHON, max_expansions=1000)
    assert path == [(1, 1), (2, 2)]                            # two diagonal steps


def test_requesting_rust_when_absent_raises() -> None:
    if _HAS_RUST:
        pytest.skip("hymeko.astar_plan IS present; this checks the absent-path error")
    with pytest.raises(RuntimeError, match="astar_plan is unavailable"):
        plan_footsteps(_wall_field(), (0, 0), (3, 0), backend=Backend.RUST)


@_skip_rust
def test_rust_backend_matches_python_cost() -> None:
    """The shared Rust engine and the Python reference return equal-cost plans (real parity)."""
    field = _wall_field()
    rust = plan_footsteps(field, (0, 0), (3, 0), backend=Backend.RUST, max_expansions=1000)
    py = plan_footsteps(field, (0, 0), (3, 0), backend=Backend.PYTHON, max_expansions=1000)
    assert rust is not None and py is not None
    assert len(rust) == len(py)                                # same optimal cost (tie-break may differ)
    assert rust[-1] == py[-1] == (3, 0)
    assert all(field.passable(c) for c in rust)


@_skip_rust
def test_rust_astar_plan_direct_and_error_propagation() -> None:
    import hymeko

    adj = {0: [(1, 1.0)], 1: [(2, 1.0)], 2: [(3, 1.0)]}         # a line 0→1→2→3
    path = hymeko.astar_plan(0, lambda n: adj.get(n, []), lambda n: n == 3, lambda n: float(3 - n), 100)
    assert path == [1, 2, 3]
    assert hymeko.astar_plan(0, lambda n: [], lambda n: n == 9, lambda n: 0.0, 100) is None  # unreachable

    def boom(_n: int):
        raise ValueError("callback boom")

    with pytest.raises(ValueError, match="callback boom"):     # a raising callback must propagate
        hymeko.astar_plan(0, boom, lambda n: False, lambda n: 0.0, 100)
