use pyo3::prelude::*;
use std::collections::HashMap;

/// Build the line-graph-style edge-to-cycle incidence matrix M_e in COO
/// form. For each query edge (u, v), M_e[query, t] = 1/|N(query)| iff
/// cycle/hyperedge t shares an endpoint with the query (one of t's
/// vertices == u or v). Self-edges optionally excluded (for k=2).
///
/// Inputs (flat representations to avoid PyO3 dict-marshaling overhead):
///   edges_u, edges_v       : (E_query,) query edge endpoints
///   csr_row_ptr            : (n_nodes+1,) vertex_to_tuples row pointer
///   csr_col_idx            : (total_nnz,) flat list of tuple indices
///   self_keys_u, self_keys_v : sorted-pair (min, max) keys for self-edge map
///   self_tuple_idx         : tuple index for each self-edge key
///   n_tuples               : T (used only for shape; not validated here)
///
/// Returns (rows, cols, vals) for torch.sparse_coo_tensor construction.
/// rows: u32 (query edge index)  cols: u32 (cycle index)  vals: f32
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
    if edges_u.len() != edges_v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u and edges_v must have the same length",
        ));
    }
    if self_keys_u.len() != self_keys_v.len()
        || self_keys_u.len() != self_tuple_idx.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "self_keys_* and self_tuple_idx must have the same length",
        ));
    }

    // Build self-edge lookup map: (min(u,v), max(u,v)) → tuple_idx
    let mut self_map: HashMap<(u32, u32), u32> = HashMap::with_capacity(self_keys_u.len());
    for i in 0..self_keys_u.len() {
        self_map.insert((self_keys_u[i], self_keys_v[i]), self_tuple_idx[i]);
    }

    let e_query = edges_u.len();
    let n_nodes = csr_row_ptr.len().saturating_sub(1);

    // Pre-size outputs. Per-edge upper bound = |adj_u| + |adj_v|; total
    // is bounded by 2 * sum(deg) but we don't know it without scanning,
    // so reserve conservatively.
    let mut rows: Vec<u32> = Vec::with_capacity(e_query * 16);
    let mut cols: Vec<u32> = Vec::with_capacity(e_query * 16);
    let mut vals: Vec<f32> = Vec::with_capacity(e_query * 16);

    // Reusable scratch buffer for merge-sort+dedup of two sorted runs.
    let mut merged: Vec<u32> = Vec::with_capacity(64);

    for ei in 0..e_query {
        let u = edges_u[ei];
        let v = edges_v[ei];
        if (u as usize) >= n_nodes || (v as usize) >= n_nodes {
            continue;
        }

        let u_start = csr_row_ptr[u as usize] as usize;
        let u_end = csr_row_ptr[u as usize + 1] as usize;
        let v_start = csr_row_ptr[v as usize] as usize;
        let v_end = csr_row_ptr[v as usize + 1] as usize;

        let adj_u = &csr_col_idx[u_start..u_end];
        let adj_v = &csr_col_idx[v_start..v_end];

        // Merge two sorted runs into `merged` with dedup.
        merged.clear();
        merged.reserve(adj_u.len() + adj_v.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < adj_u.len() && j < adj_v.len() {
            let a = adj_u[i];
            let b = adj_v[j];
            if a < b {
                if merged.last().copied() != Some(a) { merged.push(a); }
                i += 1;
            } else if b < a {
                if merged.last().copied() != Some(b) { merged.push(b); }
                j += 1;
            } else {
                if merged.last().copied() != Some(a) { merged.push(a); }
                i += 1;
                j += 1;
            }
        }
        while i < adj_u.len() {
            let a = adj_u[i];
            if merged.last().copied() != Some(a) { merged.push(a); }
            i += 1;
        }
        while j < adj_v.len() {
            let b = adj_v[j];
            if merged.last().copied() != Some(b) { merged.push(b); }
            j += 1;
        }

        // Self-edge exclusion (sorted pair lookup).
        let (k_lo, k_hi) = if u < v { (u, v) } else { (v, u) };
        if let Some(&self_t) = self_map.get(&(k_lo, k_hi)) {
            if let Ok(pos) = merged.binary_search(&self_t) {
                merged.remove(pos);
            }
        }

        let n_adj = merged.len();
        if n_adj == 0 {
            continue;
        }
        let w = 1.0_f32 / n_adj as f32;
        let ei_u32 = ei as u32;
        for &t in merged.iter() {
            rows.push(ei_u32);
            cols.push(t);
            vals.push(w);
        }
    }

    Ok((rows, cols, vals))
}
