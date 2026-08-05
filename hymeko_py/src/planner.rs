//! `astar_plan` — the shared Rust A\* (`akoire::astar`) exposed to Python.
//!
//! The search *problem* is defined by Python callables (`neighbours` / `is_goal` / `heuristic`) over
//! integer node ids; the search *loop* runs in Rust via `akoire::astar` — the same engine HOTARU's
//! HIVE-delta planner (`akoire::SearchProblem`) uses. This makes the humanoid footstep planner's
//! "mirror" of the planner framework a real code-sharing binding, not a parallel Python copy.
//!
//! This is a thin PyO3 wrapper only (CLAUDE.md §6.5 #2: algorithm logic stays in the `akoire`
//! crate; the binding crate just marshals). A callback exception is captured and re-raised after the
//! search rather than swallowed (§6.4 — no silent failure at the language boundary).
//!
//! Nodes are `i64` ids and the returned plan is the id sequence of the optimal path (excluding the
//! start), so the encoding a caller puts in a node id (e.g. a foothold cell) *is* the plan — no
//! per-edge Python object needs to cross the boundary.

use std::cell::RefCell;

use pyo3::prelude::*;

/// Run `akoire::astar` over a Python-defined implicit graph of integer nodes.
///
/// # Parameters
/// - `start`: the start node id.
/// - `neighbours(node) -> list[(next_id, cost)]`: successors, generated on demand.
/// - `is_goal(node) -> bool`: goal predicate.
/// - `heuristic(node) -> float`: remaining-cost estimate (admissible ⇒ optimal path).
/// - `max_expansions`: hard cap on expanded nodes (bounds an unbounded implicit graph).
///
/// # Returns
/// The optimal path as the list of node ids from the first step to the goal (the start is excluded),
/// or `None` if no goal is reachable within `max_expansions`.
///
/// # Errors
/// Re-raises any exception raised by a callback (captured during the search, surfaced after it).
#[pyfunction]
#[pyo3(signature = (start, neighbours, is_goal, heuristic, max_expansions))]
pub fn astar_plan(
    py: Python<'_>,
    start: i64,
    neighbours: Py<PyAny>,
    is_goal: Py<PyAny>,
    heuristic: Py<PyAny>,
    max_expansions: usize,
) -> PyResult<Option<Vec<i64>>> {
    // First callback error terminates the search (neutral values below) and is re-raised after it.
    let err: RefCell<Option<PyErr>> = RefCell::new(None);
    let nb = neighbours.bind(py);
    let ig = is_goal.bind(py);
    let hu = heuristic.bind(py);

    // The edge label is the next node id, so `AstarResult.actions` is exactly the id path.
    let neighbours_cl = |n: &i64| -> Vec<(i64, i64, f64)> {
        if err.borrow().is_some() {
            return Vec::new();
        }
        match nb.call1((*n,)).and_then(|r| r.extract::<Vec<(i64, f64)>>()) {
            Ok(v) => v
                .into_iter()
                .map(|(next, cost)| (next, next, cost))
                .collect(),
            Err(e) => {
                *err.borrow_mut() = Some(e);
                Vec::new()
            }
        }
    };
    let is_goal_cl = |n: &i64| -> bool {
        if err.borrow().is_some() {
            return false;
        }
        match ig.call1((*n,)).and_then(|r| r.extract::<bool>()) {
            Ok(b) => b,
            Err(e) => {
                *err.borrow_mut() = Some(e);
                false
            }
        }
    };
    let heuristic_cl = |n: &i64| -> f64 {
        if err.borrow().is_some() {
            return 0.0;
        }
        match hu.call1((*n,)).and_then(|r| r.extract::<f64>()) {
            Ok(h) => h,
            Err(e) => {
                *err.borrow_mut() = Some(e);
                0.0
            }
        }
    };

    let result = akoire::astar(
        start,
        neighbours_cl,
        is_goal_cl,
        heuristic_cl,
        max_expansions,
    );
    if let Some(e) = err.into_inner() {
        return Err(e);
    }
    Ok(result.actions)
}
