//! PyO3 wrapper around [`hymeko_graph::incidence::build_edge_incidence_vertex_adj`].
//!
//! 2026-05-22 split (CLAUDE.md §6.5 anti-pattern #2): the sorted-merge
//! CSR algorithm now lives in `hymeko_graph::incidence`.  This file
//! is the thin PyO3 boundary: it adapts `Vec<u32>` Python arguments to
//! the slice-based algorithm entry point and returns the COO triplet
//! as a tuple `(rows, cols, vals)` ready for `torch.sparse_coo_tensor`.

use pyo3::prelude::*;

use hymeko_graph::incidence::build_edge_incidence_vertex_adj;

/// Build the line-graph-style edge-to-cycle incidence matrix `M_e` in
/// COO form.  See `hymeko_graph::incidence` for the algorithm and
/// contract; the Python-side signature is unchanged from the
/// pre-split version (flat `u32` vectors in, three `(rows, cols,
/// vals)` vectors out).
#[pyfunction]
pub fn build_edge_incidence_vertex_adj_rs(
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    csr_row_ptr: Vec<u32>,
    csr_col_idx: Vec<u32>,
    self_keys_u: Vec<u32>,
    self_keys_v: Vec<u32>,
    self_tuple_idx: Vec<u32>,
) -> PyResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let coo = build_edge_incidence_vertex_adj(
        &edges_u,
        &edges_v,
        &csr_row_ptr,
        &csr_col_idx,
        &self_keys_u,
        &self_keys_v,
        &self_tuple_idx,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((coo.rows, coo.cols, coo.vals))
}
