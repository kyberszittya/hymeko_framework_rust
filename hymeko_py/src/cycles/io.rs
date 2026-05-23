//! PyO3 ↔ numpy / hymeko_graph glue.
//!
//! `flat_to_pyarray2`, `build_signed_graph`, `batch_to_pyarray`,
//! `pick_scorer`, `build_vertex_filter`. Pure boundary code — no algorithm.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

pub(crate) use hymeko_graph::{
    balance::{BalanceMode, CartwrightHararyPruner, DavisWeakBalancePruner},
    degree_adaptive_m_v,
    enumerate_top_k_cycles_par_batched, enumerate_top_k_cycles_par_bb_batched,
    enumerate_top_k_cycles_par_entropy_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched,
    tiered_m_v_by_degree,
    topk_cycles::{
        scorers as g_scorers, BalanceScorer, FractionNegativeScorer,
        LowRootScorer, SignProductAbsScorer, TopKCyclesBatch,
    },
    vertex_filter::{AndFilter, DegreeFilter, NoFilter, TriangleFilter, VertexFilter},
    EntropyGainScorer, HybridScorer, InverseDegreeScorer, NoOpPruner,
    SignedGraph as GSignedGraph,
};

/// Materialise a flat stride-`k` `Vec<u32>` as a numpy `(n, k)` ndarray.
/// The single boundary-crossing allocation; no per-cycle Python objects.
/// Truncates to `cap` rows for reservoir / early-stop modes if oversampled.
pub(crate) fn flat_to_pyarray2<'py>(
    py: Python<'py>,
    buf: Vec<u32>,
    k: usize,
    cap: Option<usize>,
) -> PyResult<Py<PyArray2<u32>>> {
    let mut n = buf.len() / k;
    if let Some(c) = cap {
        if n > c {
            n = c;
        }
    }
    let total = n * k;
    let buf = if buf.len() == total {
        buf
    } else {
        buf[..total].to_vec()
    };
    let arr = Array2::from_shape_vec((n, k), buf).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "ndarray reshape failed: {e}"
        ))
    })?;
    Ok(arr.into_pyarray(py).unbind())
}

/// Build a [`GSignedGraph`] from python-side `(eu, ev, es)` buffers.
pub(crate) fn build_signed_graph(
    edges_u: &[u32],
    edges_v: &[u32],
    edges_s: &[i8],
    n_nodes: u32,
) -> PyResult<GSignedGraph> {
    if edges_u.len() != edges_v.len() || edges_u.len() != edges_s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u, edges_v, edges_s must have equal length",
        ));
    }
    Ok(GSignedGraph::from_parts(n_nodes, edges_u, edges_v, edges_s))
}

/// Zero-copy convert a `TopKCyclesBatch` to `(cycles_2d, scores_1d)` numpy.
pub(crate) fn batch_to_pyarray(
    py: Python<'_>,
    batch: TopKCyclesBatch,
    k_len: usize,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let n = batch.len();
    let arr = Array2::from_shape_vec((n, k_len), batch.cycles).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}"))
    })?;
    Ok((
        arr.into_pyarray(py).unbind(),
        batch.scores.into_pyarray(py).unbind(),
    ))
}

/// String-to-scorer dispatcher (kept at the PyO3 boundary; internal Rust
/// dispatch uses the BoundedScorer struct types directly — see top_k.rs).
pub(crate) fn pick_scorer(name: &str) -> Option<fn(&[u32], &[i8]) -> f64> {
    match name {
        "balance" => Some(g_scorers::balance),
        "fraction_negative" => Some(g_scorers::fraction_negative),
        "sign_product_abs" => Some(g_scorers::sign_product_abs),
        "low_root" => Some(g_scorers::low_root),
        _ => None,
    }
}

/// Vertex pre-filter constructor; matches v1 of the prefilter plan.
/// Supported kinds:
///   * `"none"`                       — all vertices
///   * `"degree"`                     — degree ≥ `min_degree`
///   * `"triangle"`                   — vertex in ≥ 1 triangle
///   * `"compose:degree,triangle"`    — AND of both
pub(crate) fn build_vertex_filter(
    filter_kind: &str,
    min_degree: u32,
) -> PyResult<Box<dyn VertexFilter>> {
    match filter_kind {
        "" | "none" | "all" => Ok(Box::new(NoFilter)),
        "degree" => Ok(Box::new(DegreeFilter { min_degree })),
        "triangle" => Ok(Box::new(TriangleFilter)),
        "compose:degree,triangle" | "compose:degree+triangle" => {
            Ok(Box::new(AndFilter(vec![
                Box::new(DegreeFilter { min_degree }),
                Box::new(TriangleFilter),
            ])))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown filter_kind '{other}'; valid: none | degree | triangle | \
             compose:degree,triangle"
        ))),
    }
}
