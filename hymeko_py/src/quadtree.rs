//! PyO3 wrapper around [`hymeko_graph::quadtree::build_quadtree`].
//!
//! After the 2026-05-22 algorithm-vs-binding split (CLAUDE.md §6.5
//! anti-pattern #2), the per-depth subdivision state machine, the
//! Forman κ kernel and the unit tests live in `hymeko_graph::quadtree`.
//! This file is the thin Python boundary: it converts the Python
//! `score_fn` callable into a Rust closure, calls into the algorithm
//! crate, and ferries the resulting [`hymeko_graph::QuadtreeAnchors`]
//! out as numpy arrays.
//!
//! Public API (unchanged from the pre-split signature):
//!
//!   `build_quadtree_rs(image_h, image_w, patch_size_initial,
//!                      patch_size_min, max_depth, max_anchors,
//!                      variance_weight, curvature_weight,
//!                      score_threshold, score_fn)`
//!
//!     → (positions: (n, 2) int64,
//!        sizes:     (n,)   int64,
//!        scales:    (n,)   int64,
//!        parent_indices: (n,) int64)
//!
//! `score_fn(positions: list[(int, int)], sizes: list[int])
//!   → list[float]` is the Python callable that returns per-anchor
//! variance scores for the current frontier (one call per depth).

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use hymeko_graph::quadtree::{build_quadtree, QuadtreeAnchors};

/// Top-level Python entry point: run the full adaptive-subdivision
/// state machine.
///
/// Returns four arrays of length `n_anchors`:
///   * `positions` of shape (n, 2)
///   * `sizes`, `scales`, `parent_indices` each shape (n,)
///
/// The `score_fn` callback is invoked once per depth (≤ `max_depth`
/// times total) with the current frontier's positions + sizes; it
/// must return a `list[float]` of length `len(positions)` containing
/// the per-anchor variance score. Curvature is added internally; the
/// combined score is thresholded by `score_threshold`.
#[pyfunction]
#[pyo3(signature = (
    image_h, image_w, patch_size_initial, patch_size_min,
    max_depth, max_anchors,
    variance_weight, curvature_weight, score_threshold,
    score_fn,
))]
#[allow(clippy::too_many_arguments)] // outer Python-kwargs surface
pub fn build_quadtree_rs<'py>(
    py: Python<'py>,
    image_h: i64,
    image_w: i64,
    patch_size_initial: i64,
    patch_size_min: i64,
    max_depth: i64,
    max_anchors: i64,
    variance_weight: f32,
    curvature_weight: f32,
    score_threshold: f32,
    score_fn: Py<PyAny>,
) -> PyResult<(
    Py<PyArray2<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
)> {
    // Adapt the Python callable into the closure type the algorithm
    // crate expects.  Errors from the Python side are captured into
    // `closure_err` and surfaced after the algorithm returns; we
    // can't propagate `Result` through `FnMut(...) -> Vec<f32>` so the
    // closure returns an empty Vec on failure and we check after.
    let mut closure_err: Option<PyErr> = None;
    let anchors: QuadtreeAnchors = {
        let score_fn_ref = &score_fn;
        let closure_err_ref = &mut closure_err;
        let closure = |frontier_positions: &[(i64, i64)],
                       frontier_sizes: &[i64]|
         -> Vec<f32> {
            if closure_err_ref.is_some() {
                return Vec::new();
            }
            match call_score_fn(py, score_fn_ref, frontier_positions, frontier_sizes) {
                Ok(v) => {
                    if v.len() != frontier_positions.len() {
                        *closure_err_ref = Some(pyo3::exceptions::PyValueError::new_err(
                            format!(
                                "score_fn returned {} entries for a frontier of {}",
                                v.len(),
                                frontier_positions.len(),
                            ),
                        ));
                        return vec![0.0; frontier_positions.len()];
                    }
                    v
                }
                Err(e) => {
                    *closure_err_ref = Some(e);
                    vec![0.0; frontier_positions.len()]
                }
            }
        };
        build_quadtree(
            image_h,
            image_w,
            patch_size_initial,
            patch_size_min,
            max_depth,
            max_anchors,
            variance_weight,
            curvature_weight,
            score_threshold,
            closure,
        )
    };
    if let Some(e) = closure_err {
        return Err(e);
    }

    let QuadtreeAnchors {
        positions,
        sizes,
        scales,
        parent_indices,
    } = anchors;
    let n = positions.len();
    let mut positions_flat: Vec<i64> = Vec::with_capacity(n * 2);
    for &(rr, cc) in &positions {
        positions_flat.push(rr);
        positions_flat.push(cc);
    }
    let positions_arr = ndarray::Array2::from_shape_vec((n, 2), positions_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(
            format!("positions shape error: {e}"),
        ))?;

    Ok((
        positions_arr.into_pyarray(py).unbind(),
        sizes.into_pyarray(py).unbind(),
        scales.into_pyarray(py).unbind(),
        parent_indices.into_pyarray(py).unbind(),
    ))
}

/// Invoke the Python `score_fn(positions, sizes) -> list[float]`.
fn call_score_fn(
    py: Python<'_>,
    score_fn: &Py<PyAny>,
    frontier_positions: &[(i64, i64)],
    frontier_sizes: &[i64],
) -> PyResult<Vec<f32>> {
    let py_positions = PyList::empty(py);
    for &(rr, cc) in frontier_positions {
        let tup = (rr, cc).into_pyobject(py)?;
        py_positions.append(tup)?;
    }
    let py_sizes = PyList::new(py, frontier_sizes)?;
    let result = score_fn.call1(py, (py_positions, py_sizes))?;
    result.extract::<Vec<f32>>(py)
}
